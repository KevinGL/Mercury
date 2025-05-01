#include "Mercury.h"

Mercury::ChatBot::ChatBot()
{
    std::ifstream file("./Path.ini");

    if(file)
    {
        while(1)
        {
            std::string line;

            if(!getline(file, line))
            {
                break;
            }

            if(line.find("Path=") == 0)
            {
                path = line;
                path.erase(0, path.find("=") + 1);
            }
        }

        file.close();
    }

    tokenizer.loadDatas(path);
}

void Mercury::ChatBot::learn()
{
    tokenizer.learn(path);

    std::cout << "Mercury : Learning ok ..." << std::endl;
}

void Mercury::ChatBot::prompt(const std::wstring text)
{
    std::vector<unsigned int> encoded = tokenizer.encode(text);

    for(const unsigned int e : encoded)
    {
        std::cout << e << " ";
    }

    std::cout << std::endl;

    std::wcout << tokenizer.decode(encoded) << std::endl;
}
