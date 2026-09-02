// Minimal reader for config.yaml: "section:" headers, indented "key: value"
// scalars, one inline "[a, b, c]" list. Not a general YAML parser.
#pragma once

#include <map>
#include <string>
#include <vector>

namespace cattrack {

class Config {
public:
    // Throws std::runtime_error if the file is missing.
    static Config load(const std::string& path);

    bool has(const std::string& section, const std::string& key) const;
    std::string get_string(const std::string& section, const std::string& key,
                           const std::string& def = "") const;
    int get_int(const std::string& section, const std::string& key, int def = 0) const;
    double get_double(const std::string& section, const std::string& key,
                      double def = 0.0) const;
    bool get_bool(const std::string& section, const std::string& key, bool def = false) const;
    std::vector<double> get_doubles(const std::string& section,
                                    const std::string& key) const;

private:
    std::map<std::string, std::map<std::string, std::string>> data_;
};

}  // namespace cattrack
