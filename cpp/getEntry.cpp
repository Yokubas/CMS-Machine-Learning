#include <TChain.h>
#include <TH1F.h>
#include <TFile.h>
#include <TTree.h>

#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include <numeric> 
#include <algorithm>

using namespace std;

TChain* createChainFromFileList(const char* txtFileName, int maxFiles = -1) {
    TChain* chain = new TChain("Events");
    ifstream infile(txtFileName);
    string line;
    int count = 0;
    while (getline(infile, line)) {
        if (maxFiles != -1 && count >= maxFiles) break;
        chain->Add(line.c_str());
        count++;
    }
    return chain;
}

int main() {

    TChain* tree = createChainFromFileList("../data/real/CMS_Run2016H_DoubleEG_NANOAOD_UL2016_MiniAODv2_NanoAODv9-v1_100000_file_index.txt", 1);

    cout << "Entries in real data: " << tree->GetEntries() << endl;

    return 0;
}