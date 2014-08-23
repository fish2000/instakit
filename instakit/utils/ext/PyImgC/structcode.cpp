
#include <stdio.h>
#include <iostream>
#include <utility>
#include <vector>
#include <map>
#include <string>
using namespace std;

//#define as ,
//#define foreach(str, c) for (const char *c = (const char *)str; *c; c++)
#define foreach(str, c) for (char &c : str)

int main(void) {
    
    map<string, char> native_map = {
        {"?", '?'},
        {"b", 'b'},
        {"B", 'B'},
        {"h", 'h'},
        {"H", 'H'},
        {"i", 'i'},
        {"I", 'I'},
        {"l", 'l'},
        {"L", 'L'},
        {"q", 'q'},
        {"Q", 'Q'},
        {"e", 'e'},
        {"f", 'f'},
        {"d", 'd'},
        {"g", 'g'},
        {"Zf", 'F'},
        {"Zd", 'D'},
        {"Zg", 'G'},
        {"s", 'S'},
        {"w", 'U'},
        {"O", 'O'},
        {"x", 'V'}, /// padding
    };
    
    
    string most = string("?bhilqefdgswxBHILQO");
    int i = 0;
    foreach (most, chr) {
        //printf("NATIVE: %c -> %c\n", chr, native_map[string(&chr)]);
        string key = string(&chr).substr(0, 1);
        //string idx = string(&most[i]);
        //string key = string(idx)[0];
        cout << "NATIVE: " << key << " -> " << native_map[key] << "\n";
        i++;
    }
    
    /*
    int i = 0;
    foreach ("YO DOGG", chr) {
        printf("CHAR AT %i: %c\n", i, *chr);
        i++;
    }
    */
    
    printf("\n");
    return 0;
}