// Compiler: g++ 15.2.0
#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cmath>
#include <compare>
#include <cstddef>
#include <format>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>



// A point in 2D space; printed as "x y"
struct Point2D {
    double x{0.0};
    double y{0.0};

    auto operator<=>(const Point2D&) const = default;

    friend std::ostream& operator<<(std::ostream& os, const Point2D& p) {
        return os << p.x << ' ' << p.y;
    }

    bool is_finite() const noexcept {
        return std::isfinite(x) && std::isfinite(y);
    }
};

// Affine transform f(p) = M * p + v
class AffineTransform {
private:
    std::array<std::array<double, 2>, 2> m{{{1.0, 0.0}, {0.0, 1.0}}};  // identity by default
    Point2D v{0.0, 0.0};

public:
    constexpr AffineTransform() = default;

    AffineTransform(double m00, double m01, double m10, double m11, Point2D v)
        : m{{{m00, m01}, {m10, m11}}}, v{v} {}

    constexpr Point2D operator()(const Point2D& p) const noexcept {
        return Point2D{
            m[0][0] * p.x + m[0][1] * p.y + v.x,
            m[1][0] * p.x + m[1][1] * p.y + v.y
        };
    }

    auto operator<=>(const AffineTransform&) const = default;
};

// Function object: each call applies a uniformly chosen transform of the given set
class RandomAffineTransformer {
private:
    std::vector<AffineTransform> transforms_;
    std::mt19937_64 rng_;
    std::uniform_int_distribution<std::size_t> dist_;

    static std::mt19937_64 make_rng() {
        std::random_device rd;
        std::seed_seq seed{rd(), rd(), rd(), rd()};
        return std::mt19937_64(seed);
    }

public:
    explicit RandomAffineTransformer(std::vector<AffineTransform> transforms)
        : transforms_(std::move(transforms)), rng_(make_rng()) {
        if (transforms_.empty()) {
            throw std::invalid_argument("Transform list cannot be empty.");
        }
        dist_ = std::uniform_int_distribution<std::size_t>(0, transforms_.size() - 1);
    }

    Point2D operator()(const Point2D& p) {
        return transforms_[dist_(rng_)](p);
    }
};


struct SimulationConfig {
    std::size_t n{0};
    Point2D start_point;
    std::vector<AffineTransform> transforms;
};

constexpr std::size_t kMaxPoints = 1'000'000'000;

// Reads a text file line by line, skipping blank lines and counting line numbers
class LineReader {
private:
    std::ifstream file_;
    std::size_t line_num_{0};

public:
    explicit LineReader(const std::string& path) : file_(path) {
        if (!file_.is_open()) {
            throw std::runtime_error(std::format("Failed to open file at path: {}", path));
        }
    }

    // Next non-blank line, or nullopt at the end of the file
    std::optional<std::string> next() {
        std::string line;
        while (std::getline(file_, line)) {
            ++line_num_;
            const bool blank = std::ranges::all_of(
                line, [](unsigned char ch) { return std::isspace(ch) != 0; });
            if (!blank) {
                return line;
            }
        }
        return std::nullopt;
    }

    std::size_t line_number() const noexcept { return line_num_; }
};

namespace parsing {

std::vector<std::string_view> tokenize(std::string_view line) {
    std::vector<std::string_view> tokens;
    const auto is_space = [](unsigned char ch) { return std::isspace(ch) != 0; };
    std::size_t i = 0;
    while (i < line.size()) {
        while (i < line.size() && is_space(line[i])) ++i;
        const std::size_t begin = i;
        while (i < line.size() && !is_space(line[i])) ++i;
        if (i > begin) tokens.push_back(line.substr(begin, i - begin));
    }
    return tokens;
}

// Whole token must be a finite real number
double to_double(std::string_view token, std::size_t line_num) {
    std::string_view digits = token;
    if (digits.size() > 1 && digits.front() == '+' && digits[1] != '-' && digits[1] != '+') {
        digits.remove_prefix(1);  // from_chars does not accept a leading '+'
    }
    double value = 0.0;
    const auto [ptr, ec] = std::from_chars(digits.data(), digits.data() + digits.size(), value);
    if (ec != std::errc{} || ptr != digits.data() + digits.size() || !std::isfinite(value)) {
        throw std::runtime_error(std::format(
            "Line {}: '{}' is not a valid finite real number.", line_num, token));
    }
    return value;
}

// Whole token must be a non-negative integer that fits into size_t
std::size_t to_natural(std::string_view token, std::size_t line_num) {
    std::size_t value = 0;
    const auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), value);
    if (ec != std::errc{} || ptr != token.data() + token.size()) {
        throw std::runtime_error(std::format(
            "Line {}: '{}' is not a valid natural number.", line_num, token));
    }
    return value;
}

// Parses a line with exactly N real numbers
template <std::size_t N>
std::array<double, N> to_doubles(const std::string& line, std::size_t line_num,
                                 std::string_view what) {
    const auto tokens = tokenize(line);
    if (tokens.size() != N) {
        throw std::runtime_error(std::format(
            "Line {}: expected {} values ({}), found {}.", line_num, N, what, tokens.size()));
    }
    std::array<double, N> values{};
    std::ranges::transform(tokens, values.begin(),
                           [line_num](std::string_view t) { return to_double(t, line_num); });
    return values;
}

} 

std::size_t read_point_count(LineReader& reader) {
    const auto line = reader.next();
    if (!line) {
        throw std::runtime_error("File is empty: value of n was not found.");
    }
    const auto tokens = parsing::tokenize(*line);
    if (tokens.size() != 1) {
        throw std::runtime_error(std::format(
            "Line {}: expected a single natural number n.", reader.line_number()));
    }
    const std::size_t n = parsing::to_natural(tokens[0], reader.line_number());
    if (n == 0 || n > kMaxPoints) {
        throw std::runtime_error(std::format(
            "Line {}: n must be in range 1..{}.", reader.line_number(), kMaxPoints));
    }
    return n;
}

Point2D read_start_point(LineReader& reader) {
    const auto line = reader.next();
    if (!line) {
        throw std::runtime_error("Starting point coordinates were not found.");
    }
    const auto [x, y] = parsing::to_doubles<2>(*line, reader.line_number(), "x y");
    return Point2D{x, y};
}

AffineTransform parse_transform(const std::string& line, std::size_t line_num) {
    const auto v = parsing::to_doubles<6>(line, line_num, "m00 m01 m10 m11 vx vy");
    return AffineTransform{v[0], v[1], v[2], v[3], Point2D{v[4], v[5]}};
}

SimulationConfig load_configuration(const std::string& input_path) {
    LineReader reader(input_path);

    SimulationConfig config;
    config.n = read_point_count(reader);
    config.start_point = read_start_point(reader);
    while (const auto line = reader.next()) {
        config.transforms.push_back(parse_transform(*line, reader.line_number()));
    }
    if (config.transforms.empty()) {
        throw std::runtime_error("No affine transforms found in the file.");
    }
    return config;
}



// Writes n points (the starting point first) to the file, one "x y" per line
void generate_points_file(const SimulationConfig& config, const std::string& output_path) {
    std::ofstream out(output_path);
    if (!out.is_open()) {
        throw std::runtime_error(std::format("Failed to create output file: {}", output_path));
    }
    out << std::fixed << std::setprecision(10);

    RandomAffineTransformer transformer(config.transforms);
    Point2D current = config.start_point;
    std::size_t index = 0;

    // Returns the current point, then advances to the next one
    auto next_point = [&]() -> Point2D {
        const Point2D result = current;
        current = transformer(current);
        if (!current.is_finite()) {
            throw std::runtime_error(std::format("Coordinates overflowed after point number {}.", index + 1));
        }
        index++;
        return result;
    };

    std::generate_n(std::ostream_iterator<Point2D>(out, "\n"), config.n, next_point);

    out.flush();
    if (!out) {
        throw std::runtime_error(std::format("Failed to write to output file: {}", output_path));
    }
}


int main(int argc, char* argv[]) {
    if (argc > 3) {
        std::cerr << std::format("Usage: {} [input_file] [output_file]\n", argv[0]);
        return 1;
    }
    const std::string input_file = (argc > 1) ? argv[1] : "input.txt";
    const std::string output_file = (argc > 2) ? argv[2] : "output.txt";

    try {
        std::cout << std::format("Reading data from '{}'...\n", input_file);
        const auto config = load_configuration(input_file);

        std::cout << std::format("Configuration successfully loaded:\n"
                                 " - Number of points: {}\n"
                                 " - Starting point: ({}, {})\n"
                                 " - Number of transforms: {}\n",
                                 config.n, config.start_point.x, config.start_point.y,
                                 config.transforms.size());

        std::cout << std::format("Generating and writing to '{}'...\n", output_file);
        generate_points_file(config, output_file);

        std::cout << "Execution completed successfully!\n";
    } catch (const std::exception& e) {
        std::cerr << std::format("\n[ERROR]: {}\n", e.what());
        return 1;
    }
    return 0;
}
