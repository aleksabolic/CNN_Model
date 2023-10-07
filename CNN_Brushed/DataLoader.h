#pragma once

#include <functional>

template <class typeX, class typeY>
class DataLoader
{
public:
    typeX xData;
    typeY yData;
    int batchSize;
    int inputSize;
    int sampleCount;

    DataLoader(typeX x, typeY y, int batchSize);

    void LoadData(std::function<void(typeX&, typeY&)> callback);
};

template <class typeX, class typeY>
DataLoader<typeX, typeY>::DataLoader(typeX x, typeY y, int batchSize) : xData(x), yData(y), batchSize(batchSize) {
    sampleCount = x.size();
    inputSize = x[0].size();
}

template <class typeX, class typeY>
void DataLoader<typeX, typeY>::LoadData(std::function<void(typeX&, typeY&)> dataCallback)
{
    int currIndex = 0;
    while (currIndex < sampleCount - batchSize) {
        typeX xBatch(xData.begin() + currIndex, xData.begin() + currIndex + batchSize);
        typeY yBatch(yData.begin() + currIndex, yData.begin() + currIndex + batchSize);

        dataCallback(xBatch, yBatch);

        currIndex += batchSize;
    }

    // if sampleCount % batchSize != 0
    if (currIndex + 1 + batchSize != sampleCount) {
        typeX xBatch(xData.end() - batchSize, xData.end());
        typeY yBatch(yData.end() - batchSize, yData.end());

        dataCallback(xBatch, yBatch);
    }
}
