#include <TChain.h>
#include <TH1F.h>
#include <TFile.h>

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

void fillHistogramFromTree(TTree* tree, bool isMC){
    
    tree->SetBranchStatus("*", false);
    tree->SetBranchStatus("Electron_pt", true);
    tree->SetBranchStatus("Electron_eta", true);
    tree->SetBranchStatus("Electron_phi", true);
    tree->SetBranchStatus("nElectron", true);
    tree->SetBranchStatus("Electron_mass", true);
    tree->SetBranchStatus("Electron_charge", true);
    tree->SetBranchStatus("Electron_miniPFRelIso_all", true);
    tree->SetBranchStatus("Electron_miniPFRelIso_chg", true);
    tree->SetBranchStatus("Electron_dz", true);
    tree->SetBranchStatus("Electron_dxy", true);
    tree->SetBranchStatus("Electron_ip3d", true);
    tree->SetBranchStatus("Jet_pt", true);
    tree->SetBranchStatus("Jet_eta", true);
    tree->SetBranchStatus("Jet_phi", true);
    tree->SetBranchStatus("Jet_btagDeepFlavB", true);

    Float_t genWeight = 1.0;
    UInt_t nElectron;
    // sumGenWeight = 0;

    if (isMC) {
        tree->SetBranchStatus("genWeight", true);
        tree->SetBranchAddress("genWeight", &genWeight);    
    }
    tree->SetBranchAddress("nElectron", &nElectron);

    // TFile outFile(outFileName, "RECREATE");
    // TTree* newTree = new TTree("Events", "Selected branches");


    for (int iEntry = 0; tree->LoadTree(iEntry) >= 0; ++iEntry){    
        
        tree->GetEntry(iEntry);
        cout << "Entry " << iEntry << " nElectron = " << nElectron << endl;
        
        // newTree->Write(); 
        // outFile.Close();

    }
}

int main() {
    // TChain* tree = createChainFromFileList("../data/real/CMS_Run2016H_DoubleEG_NANOAOD_UL2016_MiniAODv2_NanoAODv9-v1_100000_file_index.txt", 1);  
    // TChain* mcTree = createChainFromFileList("../data/raw/signal/CMS_mc_RunIISummer20UL16NanoAODv9_DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_NANOAODSIM_106X_mcRun2_asymptotic_v17-v1_30000_file_index.txt", 1);  
    TChain* mcTreeLow = createChainFromFileList("../data/raw/signal/CMS_mc_RunIISummer20UL16NanoAODv9_DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8_NANOAODSIM_106X_mcRun2_asymptotic_v17-v1_2520000_file_index.txt", 1);

    fillHistogramFromTree(mcTreeLow, true);
    // Print number of entries
    // cout << "tree entries: " << tree->GetEntries() << endl;
    // cout << "mcTree entries: " << mcTree->GetEntries() << endl;
    cout << "mcTreeLow entries: " << mcTreeLow->GetEntries() << endl;

    return 0;
}