#ifndef PyImgC_STRUCTCODE_H
#define PyImgC_STRUCTCODE_H

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
    
    string byteorder = "@";
    string code = "xxx";
    size_t itemsize = 1;
    size_t multiplier = 1;
    vector<int> shape = {0};
    const vector<int> noshape = shape;
    
    while (true) {
        if (typecode.size() == 0) {
            if (code == "xxx") {
                break;
            }
            string dtypechar;
            string multiplierstr = "";
            if (multiplier > 1) {
                multiplierstr = to_string(multiplier);
            }
            
            if (byteorder == "@" || byteorder == "^") {
                try {
                    dtypechar = multiplierstr + typecodemaps::native.at(code);
                } catch (const out_of_range &err) {
                    cerr    << ">>> Native typecode symbol not found: "
                            << code << "\n>>> Exception message: "
                            << err.what() << "\n";
                    break;
                }
            } else {
                try {
                    dtypechar = byteorder + multiplierstr + typecodemaps::standard.at(code);
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
                dtypechar += outstream.str();
            }
            
            tokens.push_back(dtypechar);
            break;
        }
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
                if (code == "xxx") {
                    /// it's a multiplier
                    multiplier = (size_t)stol(numstr);
                } else {
                    /// it's an item size
                    itemsize = (size_t)stol(numstr);
                }
                typecode.erase(0, siz);
                if (numstr.back() == '(') {
                    typecode = "(" + typecode;
                }
            }
            break;
            default:
            size_t codelen = 1;
            if (typecode.substr(0, 1) == "Z") {
                /// add next character
                codelen++;
            }
            
            code = typecode.substr(0, codelen);
            typecode.erase(0, codelen);
            break;
        }
    }
    
    if (toplevel) {
        
    }
    
    return tokens;
}

#endif