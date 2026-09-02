#include "cattrack/config.h"

#include <fstream>
#include <stdexcept>

namespace cattrack {

namespace {

std::string strip(const std::string& s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    const auto last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

// no '#' appears inside a value in this file, so this is safe
std::string decomment(const std::string& s) {
    const auto h = s.find('#');
    return h == std::string::npos ? s : s.substr(0, h);
}

std::string unquote(const std::string& s) {
    if (s.size() >= 2 && (s.front() == '"' || s.front() == '\'') && s.back() == s.front())
        return s.substr(1, s.size() - 2);
    return s;
}

}  // namespace

Config Config::load(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cattrack::Config: cannot open " + path);

    Config cfg;
    std::string section;
    std::string line;
    while (std::getline(in, line)) {
        const bool indented = !line.empty() && (line[0] == ' ' || line[0] == '\t');
        const std::string body = strip(decomment(line));
        if (body.empty()) continue;

        const auto colon = body.find(':');
        if (colon == std::string::npos) continue;

        const std::string key = strip(body.substr(0, colon));
        const std::string value = strip(body.substr(colon + 1));

        if (!indented && value.empty()) {
            section = key;
        } else if (!section.empty()) {
            cfg.data_[section][key] = unquote(value);
        }
    }
    return cfg;
}

bool Config::has(const std::string& section, const std::string& key) const {
    const auto s = data_.find(section);
    return s != data_.end() && s->second.count(key) > 0;
}

std::string Config::get_string(const std::string& section, const std::string& key,
                               const std::string& def) const {
    const auto s = data_.find(section);
    if (s == data_.end()) return def;
    const auto k = s->second.find(key);
    return k == s->second.end() ? def : k->second;
}

int Config::get_int(const std::string& section, const std::string& key, int def) const {
    return has(section, key) ? std::stoi(get_string(section, key)) : def;
}

double Config::get_double(const std::string& section, const std::string& key,
                          double def) const {
    return has(section, key) ? std::stod(get_string(section, key)) : def;
}

bool Config::get_bool(const std::string& section, const std::string& key, bool def) const {
    if (!has(section, key)) return def;
    const std::string v = get_string(section, key);
    return v == "true" || v == "True" || v == "yes" || v == "1";
}

std::vector<double> Config::get_doubles(const std::string& section,
                                        const std::string& key) const {
    std::vector<double> out;
    if (!has(section, key)) return out;
    std::string v = get_string(section, key);
    for (char& c : v)
        if (c == '[' || c == ']' || c == ',') c = ' ';
    std::size_t pos = 0;
    while (pos < v.size()) {
        std::size_t consumed = 0;
        try {
            out.push_back(std::stod(v.substr(pos), &consumed));
        } catch (const std::exception&) {
            break;
        }
        pos += consumed;
        while (pos < v.size() && v[pos] == ' ') ++pos;
    }
    return out;
}

}  // namespace cattrack
