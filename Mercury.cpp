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
    embedder.loadDatas(path);
}

void Mercury::ChatBot::learn()
{
    tokenizer.learn(path);
    std::cout << "Mercury : Learning tokenizer ok ..." << std::endl;

    embedder.learn(path, tokenizer);
    std::cout << "Mercury : Learning embeddings ok ..." << std::endl;
}

void Mercury::ChatBot::prompt(const std::wstring text)
{
    /*std::vector<unsigned int> encoded = tokenizer.encode(text);

    for(const unsigned int e : encoded)
    {
        std::cout << e << " ";
    }

    std::cout << std::endl;

    std::wcout << tokenizer.decode(encoded) << std::endl;*/

    std::vector<unsigned int> encoded = tokenizer.encode(text);
    const unsigned int lastToken = encoded[encoded.size() - 1];

    std::wcout << lastToken << L" (" << tokenizer.getIds()[lastToken] << L")" << std::endl;

    std::vector<float> embedding = embedder.getEmbeddings()[lastToken];

    embedder.getPredNetwork().feedForward(embedding);

    std::vector<unsigned int> arrayIds = tokenizer.getArrayIds();

    std::vector<float> vectorProba;
    softmax(embedder.getPredNetwork().getLayer("output"), vectorProba);
    const size_t indexMax = getIndexMax(vectorProba);
    std::cout << indexMax << " (" << vectorProba.size() - 1 << ")" << std::endl;
    //const unsigned int tokenPredicted = arrayIds[indexMax];

    //std::wcout << L"=> " << tokenPredicted << L" (" << tokenizer.getIds()[tokenPredicted] << std::endl;
}
