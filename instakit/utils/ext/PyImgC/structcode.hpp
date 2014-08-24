#ifndef PyImgC_STRUCTCODE_H
#define PyImgC_STRUCTCODE_H

#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <map>
using namespace std;

#define foreach(str, c) for (char &c : str)

struct typecodemaps {
    
    static map<string, string> init_byteorder() {
        map<string, string> _byteorder_map = {
            {"@", "="},
            {"=", "="},
            {"<", "<"},
            {">", ">"},
            {"^", "="},
            {"!", ">"},
        };
        return _byteorder_map;
    }
    
    static map<string, string> init_native() {
        map<string, string> _native_map = {
            {"?", "?"},
            {"b", "b"},
            {"B", "B"},
            {"h", "h"},
            {"H", "H"},
            {"i", "i"},
            {"I", "I"},
            {"l", "l"},
            {"L", "L"},
            {"q", "q"},
            {"Q", "Q"},
            {"e", "e"},
            {"f", "f"},
            {"d", "d"},
            {"g", "g"}, 
            {"Zf", "F"},
            {"Zd", "D"},
            {"Zg", "G"},
            {"s", "S"},
            {"w", "U"},
            {"O", "O"},
            {"x", "V"}, /// padding
        };
        return _native_map;
    }
    
    static map<string, string> init_standard() {
        map<string, string> _standard_map = {
            {"?", "?"},
            {"b", "b"},
            {"B", "B"},
            {"h", "i2"},
            {"H", "u2"},
            {"i", "i4"},
            {"I", "u4"},
            {"l", "i4"},
            {"L", "u4"},
            {"q", "i8"},
            {"Q", "u8"},
            {"e", "f2"},
            {"f", "f"},
            {"d", "d"},
            {"Zf", "F"},
            {"Zd", "D"},
            {"s", "S"},
            {"w", "U"},
            {"O", "O"},
            {"x", "V"}, /// padding
        };
        return _standard_map;
    }
    
    static const map<string, string> byteorder;
    static const map<string, string> native;
    static const map<string, string> standard;
};

const map<string, string> typecodemaps::byteorder = typecodemaps::init_byteorder();
const map<string, string> typecodemaps::native = typecodemaps::init_native();
const map<string, string> typecodemaps::standard = typecodemaps::init_standard();

struct field_namer {
    int idx;
    vector<string> field_name_vector;
    field_namer():idx(0) {}
    int next() { return idx++; }
    void add(string field_name) { field_name_vector.push_back(field_name); }
    bool has(string field_name) {
        for (auto fn = begin(field_name_vector); fn != end(field_name_vector); ++fn) {
            if (string(*fn) == field_name) {
                return true;
            }
        }
        return false;
    }
    string operator()() {
        char str[255];
        while (true) {
            sprintf(str, "f%i", next());
            string dummy_name = string(str);
            if (!has(dummy_name)) {
                add(dummy_name);
                return dummy_name;
            }
        }
    }
};

vector<int> parse_shape(string shapecode) {
    //cerr << "Shape string: " << shapecode << "\n";
    string segment;
    vector<int> shape_elems;
    while (shapecode.find(",", 0) != string::npos) {
        size_t pos = shapecode.find(",", 0);
        segment = shapecode.substr(0, pos);
        shapecode.erase(0, pos+1);
        shape_elems.push_back(stoi(segment));
    }
    shape_elems.push_back(stoi(shapecode));
    return shape_elems;
}

vector<string> parse(string typecode, bool toplevel=true) {
    vector<string> tokens;
    vector<pair<string, string>> fields;
    field_namer field_names;
    
    string byteorder = "@";
    size_t itemsize = 1;
    vector<int> shape = {0};
    const vector<int> noshape = shape;
    
    //cerr << "Typecode string: " << typecode << "\n";
    
    while (true) {
        if (typecode.size() == 0) { break; }
        switch (typecode[0]) {
            case '{': {
                typecode.erase(0, 1);
                int pos = 1;
                size_t siz;
                for (siz = 0; pos && (siz != typecode.size()); ++siz) {
                    if (typecode[siz] == '{') { ++pos; }
                    if (typecode[siz] == '}') { --pos; }
                }
                if (pos) { break; } /// too many open-brackets
                string temp = typecode.substr(0, siz-1);
                vector<string> temp_tokens;
                temp_tokens = parse(temp, toplevel=false);
                typecode.erase(0, siz+1);
                for (size_t idx = 0; idx < temp_tokens.size(); ++idx) {
                    tokens.push_back(temp_tokens[idx]);
                }
            }
            break;
            case '(': {
                typecode.erase(0, 1);
                int pos = 1;
                size_t siz;
                for (siz = 0; pos && (siz != typecode.size()); ++siz) {
                    if (typecode[siz] == '(') { ++pos; }
                    if (typecode[siz] == ')') { --pos; }
                }
                if (pos) { break; } /// too many open-parens
                string shapestr = typecode.substr(0, siz-1);
                shape = parse_shape(shapestr);
                typecode.erase(0, siz);
                //cerr << "Typecode after shape erasure: " << typecode << "\n";
            }
            break;
            case '*':
            {
                typecode.erase(0, 1);
                size_t pos = typecode.find("*", 0);
                string explicit_name = typecode.substr(0, pos);
                field_names.add(explicit_name);
                typecode.erase(0, pos+1);
            }
            break;
            case '@':
            case '=':
            case '<':
            case '>':
            case '^':
            case '!':
            {
                try {
                    byteorder = typecodemaps::byteorder.at(typecode.substr(0, 1));
                } catch (const out_of_range &err) {
                    cerr    << ">>> Byte order symbol not found: "
                            << typecode.substr(0, 1) << "\n>>> Exception message: "
                            << err.what() << "\n";
                }
                typecode.erase(0, 1);
                tokens.push_back(byteorder);
            }    
            break;
            case ' ':
            {
                /// NOP
                typecode.erase(0, 1);
            }
            break;
            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9':
            {
                size_t siz;
                char digit;
                int still_digits = 1;
                for (siz = 0; still_digits && (siz < typecode.size()); siz++) {
                    digit = typecode.c_str()[siz];
                    still_digits = isdigit(digit) && digit != '(';
                }
                string numstr = string(typecode.substr(0, siz));
                itemsize = (size_t)stol(numstr);
                typecode.erase(0, siz);
                if (!isdigit(numstr.back())) {
                    typecode = string(&numstr.back()) + typecode;
                }
                //cerr << "Typecode after number erasure: " << typecode << "\n";
            }
            break;
            default:
            size_t codelen = 1;
            string code = "";
            string name = "";
            string dtypechar = "";
            
            if (typecode.substr(0, codelen) == "Z") {
                /// add next character
                codelen++;
            }
            
            code += string(typecode.substr(0, codelen));
            typecode.erase(0, codelen);
            
            /// The program segfaults itself if I don't do this:
            //ostringstream wat;
            //wat << "Typecode: " << code << "\n";
            //wat.str();
            
            /// field name
            if (typecode.substr(0, 1) == ":") {
                typecode.erase(0, 1);
                size_t pos = typecode.find(":", 0);
                name = typecode.substr(0, pos);
                field_names.add(name);
                typecode.erase(0, pos+1);
            }
            
            name = name.size() ? name : field_names();
            
            if (byteorder == "@" || byteorder == "^") {
                try {
                    dtypechar = typecodemaps::native.at(code);
                } catch (const out_of_range &err) {
                    cerr    << ">>> Native typecode symbol not found: "
                            << code << "\n>>> Exception message: "
                            << err.what() << "\n";
                    break;
                }
            } else {
                try {
                    dtypechar = typecodemaps::standard.at(code);
                } catch (const out_of_range &err) {
                    cerr    << ">>> Standard typecode symbol not found: "
                            << code << "\n>>> Exception message: "
                            << err.what() << "\n";
                    break;
                }
            }
            
            const char last = dtypechar.back();
            if (last == 'U' || last == 'S' || last == 'V') {
                if (itemsize > 1) {
                    ostringstream outstream;
                    outstream << itemsize;
                    dtypechar += outstream.str();
                }
                itemsize = 1;
            }
            
            if (shape != noshape) {
                ostringstream outstream;
                outstream << "(";
                for (auto shape_elem = begin(shape); shape_elem != end(shape); ++shape_elem) {
                    outstream << *shape_elem;
                    if (shape_elem + 1 != end(shape)) {
                        outstream << ", ";
                    }
                }
                outstream << ")";
                dtypechar = outstream.str() + dtypechar;
            } else if (itemsize > 1) {
                dtypechar = to_string(itemsize) + dtypechar;
            }
            
            fields.push_back(make_pair(name, dtypechar));
            tokens.push_back(dtypechar);
            
            //byteorder = "@";
            itemsize = 1;
            shape = noshape;
            
            break;
        }
    }
    
    if (toplevel) {
        cout        << "> BYTE ORDER: " << byteorder << "\n";
        for (size_t idx = 0; idx < fields.size(); idx++) {
            cout    << "> FIELD: " << fields[idx].first
                    << " -> " << fields[idx].second
                    << "\n";
        }
    }
    
    return tokens;
}

#endif