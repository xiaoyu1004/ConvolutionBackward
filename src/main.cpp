#include "backward_filter.h"
#include "backward_data.h"

#include <map>
#include <string>
#include <vector>

int main()
{
    std::vector<std::map<std::string, int>> conv_data =
        {
            {{"n", 1}, {"c", 64}, {"h", 224}, {"w", 224}, {"oc", 64}, {"kh", 3}, {"kw", 3}, {"sh", 1}, {"sw", 1}, {"ph", 1}, {"pw", 1}, {"dh", 1}, {"dw", 1}, {"g", 1}}
            // {{"n", 32}, {"c", 64}, {"h", 224}, {"w", 224}, {"oc", 64}, {"kh", 3}, {"kw", 3}, {"sh", 1}, {"sw", 1}, {"ph", 1}, {"pw", 1}, {"dh", 1}, {"dw", 1}, {"g", 1}}
        };

    for (std::map<std::string, int> &m : conv_data)
    {
        ConvolutionBackwardFilter(m["n"], m["c"], m["h"], m["w"], m["oc"], m["kh"], m["kw"], m["sh"], m["sw"], m["ph"], m["pw"], m["dh"], m["dw"], m["g"]);
        // ConvolutionBackwardData(m["n"], m["c"], m["h"], m["w"], m["oc"], m["kh"], m["kw"], m["sh"], m["sw"], m["ph"], m["pw"], m["dh"], m["dw"], m["g"]);
    }
}