
//#include <numeric> /// for accumulate()
//#include <utility>

#include <iostream>
#include <vector>
#include <string>

#include "numpypp/structcode.hpp"
using namespace std;

int plain(void) {
    string most = string("?bhilqefdgswxBHILQO");
    for (char &chr : most) {
        string key = string(&chr).substr(0, 1);
        cout << "NATIVE: " << key << " -> " << typecodemaps::native.at(key) << "\n";
    }
    return 0;
}

int main(void) {
    vector<string> codes = {
        "B", "L", "b", "f", "Zd",
        ">Zd", ">I", ">h",
        "2B", "4L", "8b", "16f", "32Zd",
        ">2Zd", ">4I", ">8h",
        "8w", ">8w", ">(640, 480)8w", ">(640, 480)16w",
        "(640, 480)B",
        "(640, 480)B:mybytes:",
        ">(640, 480)B",
        "BBBB", ">BBBB", "BBBB:mybyte:", ">BBBB:myotherbyte:",
    };
    
    for (auto code = begin(codes); code != end(codes); ++code) {
        vector<pair<string, string>> pairvec = parse(*code);
        string dtype = "";
        for (size_t idx = 0; idx < pairvec.size(); idx++) {
            dtype += pairvec[idx].second;
        }
        cout << "CODE: " << *code << " -> " << dtype << "\n";
    }
    
    /*
    for (auto code = begin(codes); code != end(codes); ++code) {
        vector<string> vec = parse(*code);
        string dtype = accumulate(vec.begin(), vec.end(), string(""));
        cout << "CODE: " << *code << " -> " << dtype << "\n";
    }
    */
}