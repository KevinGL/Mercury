#include "Mercury.h"

void Mercury::Embedder::learn(const std::string path, Tokenizer &tokenizer)
{
    InitNetwork();

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
            const std::vector<float> embedding = embeddings[pair.first];

            size_t indexInput = 0;

            for(const float coord : embedding)
            {
                predictionNetwork.input.neurons[indexInput].output = coord;

                indexInput++;
            }

            for(size_t i = 0 ; i < MERCURY_SIZE_HIDDEN_LAYER ; i++)
            {
                predictionNetwork.hidden.neurons[i].output = 0.0f;

                for(size_t j = 0 ; j < predictionNetwork.hidden.neurons[i].weights.size() ; j++)
                {
                    const float weight = predictionNetwork.hidden.neurons[i].weights[j];
                    const float output = predictionNetwork.input.neurons[j].output;

                    predictionNetwork.hidden.neurons[i].output += weight * output;
                }

                predictionNetwork.hidden.neurons[i].output += predictionNetwork.hidden.neurons[i].bias;

                //predictionNetwork.hidden.neurons[i].output = tanh(predictionNetwork.hidden.neurons[i].output);

                //std::cout << predictionNetwork.hidden.neurons[i].output << std::endl;
            }

            size_t indexOutput = 0;

            for(size_t i = 0 ; i < MERCURY_MAX_TOKENS_OUTPUT_LAYER ; i++)
            {
                predictionNetwork.output.neurons[i].output = 0.0f;

                for(size_t j = 0 ; j < predictionNetwork.output.neurons[i].weights.size() ; j++)
                {
                    const float weight = predictionNetwork.output.neurons[i].weights[j];
                    const float output = predictionNetwork.hidden.neurons[j].output;

                    predictionNetwork.output.neurons[i].output += weight * output;
                }

                predictionNetwork.output.neurons[i].output += predictionNetwork.output.neurons[i].bias;

                //predictionNetwork.output.neurons[i].output = tanh(predictionNetwork.output.neurons[i].output);

                //std::cout << predictionNetwork.output.neurons[i].output << std::endl;

                indexOutput++;
            }

            std::vector<float> vectorProba;

            softmax(predictionNetwork.output, vectorProba);

            const size_t indexMax = getIndexMax(vectorProba);

            std::wcout << L"Input : " << pair.first << L" (" << tokenizer.getIds()[pair.first] << L")" << std::endl;
            std::wcout << L"Ouput : " << arrayIds[indexMax] << L" (" << tokenizer.getIds()[arrayIds[indexMax]] << L")" << std::endl;
            std::wcout << L"Attempted : " << pair.second << L" (" << tokenizer.getIds()[pair.second] << L")" << std::endl << std::endl;
            getch();
        }
    }
}

void Mercury::Embedder::InitNetwork()
{
    for(size_t i = 0 ; i < MERCURY_MAX_SIZE_EMBEDDINGS ; i++)
    {
        Neuron neuron;

        predictionNetwork.input.neurons.push_back(neuron);
    }

    for(size_t i = 0 ; i < MERCURY_SIZE_HIDDEN_LAYER ; i++)
    {
        Neuron neuron;

        for(size_t j = 0 ; j < MERCURY_MAX_SIZE_EMBEDDINGS ; j++)
        {
            neuron.weights.push_back((rand() % 101) / 100.0f);
        }

        neuron.bias = (rand() % 101) / 100.0f;

        predictionNetwork.hidden.neurons.push_back(neuron);
    }

    for(size_t i = 0 ; i < MERCURY_MAX_TOKENS_OUTPUT_LAYER ; i++)
    {
        Neuron neuron;

        for(size_t j = 0 ; j < MERCURY_SIZE_HIDDEN_LAYER ; j++)
        {
            neuron.weights.push_back((rand() % 101) / 100.0f);
        }

        neuron.bias = (rand() % 101) / 100.0f;

        predictionNetwork.output.neurons.push_back(neuron);
    }
}
