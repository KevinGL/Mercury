#include "Mercury.h"

void Mercury::Embedder::learn(const std::string path, Tokenizer &tokenizer)
{
    InitNetwork(tokenizer);

    std::vector<unsigned int> arrayIds = tokenizer.getArrayIds();

    for(const auto& kv : tokenizer.getTokens())
    {
        for(size_t i = 0 ; i < MERCURY_MAX_SIZE_EMBEDDINGS ; i++)
        {
            embeddings[kv.second].push_back((rand() % 201 - 100) / 100.0f);
        }
    }

    /*for(const auto& kv : embeddings)
    {
        std::cout << kv.first << " => [";

        for(const float value : kv.second)
        {
            std::cout << value << " ";
        }

        std::cout << "]" << std::endl;
    }*/

    std::vector<std::pair<unsigned int, unsigned int>> pairs;

    std::wifstream file(path + "/Mercury/Corpus2.txt");

    std::vector<std::wstring> corpusText;

    while(1)
    {
        std::wstring line;

        if(!getline(file, line))
        {
            break;
        }

        corpusText.push_back(line);
    }

    file.close();

    for(const std::wstring line : corpusText)
    {
        std::vector<unsigned int> ids = tokenizer.encode(line);

        /*std::wcout << line << std::endl;

        for(const unsigned int id : ids)
        {
            std::cout << id << std::endl;
        }*/

        if(ids.size() > 0)
        {
            std::pair<unsigned int, unsigned int> pair;

            for(size_t i = 0 ; i < ids.size() - 1 ; i++)
            {
                /*std::cout << ids[i + 0] << " => " << embeddings[ids[i + 0]].size() << std::endl;
                std::cout << ids[i + 1] << " => " << embeddings[ids[i + 1]].size() << std::endl << std::endl;*/

                pair.first = ids[i + 0];
                pair.second = ids[i + 1];

                pairs.push_back(pair);
            }
        }
    }

    size_t indexToken = 0;

    for(const std::pair<unsigned int, unsigned int> pair : pairs)
    {
        /*for(const float value : kv.second)
        {
            Neuron neuron;

            neuron.output = value;

            predictionNetwork.input.neurons.push_back(neuron);
        }*/

        //std::cout << pair.first << " " << pair.second << std::endl;

        if(pair.first != 1 && pair.second != 1)     //Case spaces
        {
            float crossEntropy;
            size_t counter = 0;

            while(1)
            {
                std::vector<float> embedding = embeddings[pair.first];

                predictionNetwork.feedForward(embedding);

                std::vector<float> vectorProba;
                softmax(predictionNetwork.getLayer("output"), vectorProba);
                const size_t indexMax = getIndexMax(vectorProba);
                const unsigned int tokenPredicted = arrayIds[indexMax];

                const int indexAttempted = indexArray(arrayIds, pair.second);

                //if(indexAttempted > -1 && indexAttempted < MERCURY_MAX_TOKENS_OUTPUT_LAYER)
                if(indexAttempted > -1 && indexAttempted < tokenizer.getTokens().size())
                {
                    std::vector<float> vectorOneHot = getVectorOneHot(indexAttempted, tokenizer.getTokens().size());
                    crossEntropy = getCrossEntropy(vectorProba, vectorOneHot, tokenizer.getTokens().size());

                    //std::cout << "Cross entropy : " << crossEntropy << std::endl;

                    //getch();

                    predictionNetwork.backPropagation(vectorProba, vectorOneHot);
                }

                if(crossEntropy < 0.05f)
                {
                    break;
                }

                counter++;
                if(counter > 100)
                {
                    break;
                }
            }

            /*std::wcout << L"Learning ok for " << tokenizer.getIds()[pair.first] << L" and " << tokenizer.getIds()[pair.second] << std::endl;
            getch();*/
        }

        indexToken++;

        std::cout << indexToken << " on " << pairs.size() << std::endl;
    }

    predictionNetwork.save(path + "/Mercury/PredictionNetwork.txt");
}

void Mercury::Embedder::InitNetwork(Mercury::Tokenizer &tokenizer)
{
    predictionNetwork.Init(tokenizer.getTokens().size());
}
