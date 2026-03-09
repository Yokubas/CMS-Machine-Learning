#include <TChain.h>
#include <fstream>
#include <string>
#include <iostream>

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
    TChain* mcTree = createChainFromFileList("../data/raw/signal/CMS_mc_RunIISummer20UL16NanoAODv9_DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_NANOAODSIM_106X_mcRun2_asymptotic_v17-v1_30000_file_index.txt", 1);  
    TChain* mcTreeLow = createChainFromFileList("../data/raw/signal/CMS_mc_RunIISummer20UL16NanoAODv9_DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8_NANOAODSIM_106X_mcRun2_asymptotic_v17-v1_2520000_file_index.txt", 1);

   
    // Print number of entries
    cout << "tree entries: " << tree->GetEntries() << endl;
    cout << "mcTree entries: " << mcTree->GetEntries() << endl;
    cout << "mcTreeLow entries: " << mcTreeLow->GetEntries() << endl;

    return 0;
}