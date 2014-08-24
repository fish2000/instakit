
#include <stdio.h>
#include <numeric>
#include <iostream>
#include <utility>
#include <vector>
#include <map>
#include <string>
#include "structcode.hpp"
using namespace std;


int plain(void) {
    string most = string("?bhilqefdgswxBHILQO");
    int i = 0;
    foreach (most, chr) {
        string key = string(&chr).substr(0, 1);
        cout << "NATIVE: " << key << " -> " << typecodemaps::native.at(key) << "\n";
        i++;
    }
    printf("\n");
    return 0;
}

int main(void) {
    vector<string> codes = {
        "B", "L", "b", "f", "Zd",
        ">Zd", ">I", ">h",
        "2B", "4L", "8b", "16f", "32Zd",
        ">2Zd", ">4I", ">8h",
        "w8", ">w8", ">w8(640, 480)", ">w16(640, 480)",
        "(640, 480)B",
        "(640, 480)B:mybytes:",
        ">(640, 480)B",
        "BBBB", ">BBBB", "BBBB:mybyte:", ">BBBB:myotherbyte:",
    };
    
    for (auto code = begin(codes); code != end(codes); ++code) {
        vector<string> vec = parse(*code);
        string dtype = accumulate(vec.begin(), vec.end(), string(""));
        cout << "CODE: " << *code << " -> " << dtype << "\n";
    }
}