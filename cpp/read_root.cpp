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

Float_t Electron_pt[100], Electron_eta[100], Electron_phi[100], Electron_mass[100];
Float_t Electron_miniPFRelIso_all[100], Electron_miniPFRelIso_chg[100];
Float_t Electron_dz[100], Electron_dxy[100], Electron_ip3d[100], Electron_charge[100];

Float_t Jet_pt[100], Jet_eta[100], Jet_phi[100], Jet_btagDeepFlavB[100];

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

void writeRootFile(TTree* tree, const char* outFileName, double& sumGenWeight, bool isMC){
    
    tree->SetBranchStatus("*", false);
    tree->SetBranchStatus("Electron_pt", true);
    tree->SetBranchStatus("Electron_eta", true);
    tree->SetBranchStatus("Electron_phi", true);
    tree->SetBranchStatus("nElectron", true);
    tree->SetBranchStatus("Electron_mass", true);

    if (!isMC){
    tree->SetBranchStatus("Electron_miniPFRelIso_all", true);
    tree->SetBranchStatus("Electron_miniPFRelIso_chg", true);
    tree->SetBranchStatus("Electron_dz", true);
    tree->SetBranchStatus("Electron_dxy", true);
    tree->SetBranchStatus("Electron_ip3d", true);

    tree->SetBranchStatus("nJet", true);
    tree->SetBranchStatus("Jet_pt", true);
    tree->SetBranchStatus("Jet_eta", true);
    tree->SetBranchStatus("Jet_phi", true);
    tree->SetBranchStatus("Jet_btagDeepFlavB", true);
    }
    
    Float_t genWeight;
    sumGenWeight = 0;
    Float_t out_genWeight;
    UInt_t nElectron;
    UInt_t nJet;
    UInt_t out_nElectron;
    
    Float_t out_Electron_pt[2], out_Electron_eta[2], out_Electron_phi[2], out_Electron_mass[2];
    Float_t out_Jet_pt[4], out_Jet_eta[4], out_Jet_phi[4], out_Jet_btagDeepFlavB[4];
    
    Float_t out_Electron_miniPFRelIso_all[2], out_Electron_miniPFRelIso_chg[2];
    
    Float_t out_Electron_dz[2], out_Electron_dxy[2], out_Electron_ip3d[2];
    
    if (isMC) {
        tree->SetBranchStatus("genWeight", true);
        tree->SetBranchAddress("genWeight", &genWeight);    
    }

    tree->SetBranchAddress("Electron_pt", &Electron_pt);
    tree->SetBranchAddress("Electron_eta", &Electron_eta);
    tree->SetBranchAddress("Electron_phi", &Electron_phi);
    tree->SetBranchAddress("nElectron", &nElectron);
    tree->SetBranchAddress("Electron_mass", Electron_mass);

    if (!isMC){
    tree->SetBranchAddress("Electron_miniPFRelIso_all", Electron_miniPFRelIso_all);
    tree->SetBranchAddress("Electron_miniPFRelIso_chg", Electron_miniPFRelIso_chg);
    tree->SetBranchAddress("Electron_dz", Electron_dz);
    tree->SetBranchAddress("Electron_dxy", Electron_dxy);
    tree->SetBranchAddress("Electron_ip3d", Electron_ip3d);

    tree->SetBranchAddress("nJet", &nJet);
    tree->SetBranchAddress("Jet_pt", Jet_pt);
    tree->SetBranchAddress("Jet_eta", Jet_eta);
    tree->SetBranchAddress("Jet_phi", Jet_phi);
    tree->SetBranchAddress("Jet_btagDeepFlavB", Jet_btagDeepFlavB);
    }

    TFile *outFile = new TFile(outFileName,"RECREATE");
    TTree *outTree = new TTree("Events","Selected branches");

    // output branches
    outTree->Branch("nElectron", &out_nElectron, "nElectron/i");
    outTree->Branch("Electron_pt", out_Electron_pt, "Electron_pt[2]/F");
    outTree->Branch("Electron_eta", out_Electron_eta, "Electron_eta[2]/F");
    outTree->Branch("Electron_phi", out_Electron_phi, "Electron_phi[2]/F");
    outTree->Branch("Electron_mass", out_Electron_mass, "Electron_mass[2]/F");
    if(isMC) outTree->Branch("genWeight", &out_genWeight, "genWeight/F");

    if (!isMC) {
    outTree->Branch("Electron_miniPFRelIso_all", out_Electron_miniPFRelIso_all, "Electron_miniPFRelIso_all[2]/F");
    outTree->Branch("Electron_miniPFRelIso_chg", out_Electron_miniPFRelIso_chg, "Electron_miniPFRelIso_chg[2]/F");
    outTree->Branch("Electron_dz", out_Electron_dz, "Electron_dz[2]/F");
    outTree->Branch("Electron_dxy", out_Electron_dxy, "Electron_dxy[2]/F");
    outTree->Branch("Electron_ip3d", out_Electron_ip3d, "Electron_ip3d[2]/F");

    outTree->Branch("Jet_pt", out_Jet_pt, "Jet_pt[4]/F");
    outTree->Branch("Jet_eta", out_Jet_eta, "Jet_eta[4]/F");
    outTree->Branch("Jet_phi", out_Jet_phi, "Jet_phi[4]/F");
    outTree->Branch("Jet_btagDeepFlavB", out_Jet_btagDeepFlavB, "Jet_btagDeepFlavB[4]/F");
    }

    for (int iEntry = 0; tree->LoadTree(iEntry) >= 0; ++iEntry){    
        
        tree->GetEntry(iEntry);

        if (isMC) sumGenWeight += genWeight;

        if (nElectron > 1){

            vector<int> twoElectrons;

            for (UInt_t i = 0; i < nElectron; ++i)
                twoElectrons.push_back(i);


            sort(twoElectrons.begin(), twoElectrons.end(), [&](int a, int b) {
                return Electron_pt[a] > Electron_pt[b];});

            int idx1 = twoElectrons[0];  // leading
            int idx2 = twoElectrons[1];  // subleading

            // copy values to output variables
            out_nElectron = 2;
            
            int electronIndices[2] = {idx1, idx2};

            for (int i = 0; i < 2; i++) {

                int id = electronIndices[i];

                out_Electron_pt[i]  = Electron_pt[id];
                out_Electron_eta[i] = Electron_eta[id];
                out_Electron_phi[i] = Electron_phi[id];
                out_Electron_mass[i] = Electron_mass[id];

                if(!isMC) {
                    out_Electron_miniPFRelIso_all[i] = Electron_miniPFRelIso_all[id];
                    out_Electron_miniPFRelIso_chg[i] = Electron_miniPFRelIso_chg[id];

                    out_Electron_dz[i] = Electron_dz[id];
                    out_Electron_dxy[i] = Electron_dxy[id];
                    out_Electron_ip3d[i] = Electron_ip3d[id];
            
                }    
            }
            
            if (isMC){ 
                out_genWeight = genWeight;
                outTree->Fill();
            }
        }
        if (!isMC){
            for (UInt_t j = 0; j < 4; j++) {
                if(j < nJet){
                    out_Jet_pt[j] = Jet_pt[j];
                    out_Jet_eta[j] = Jet_eta[j];
                    out_Jet_phi[j] = Jet_phi[j];
                    out_Jet_btagDeepFlavB[j] = Jet_btagDeepFlavB[j];
                }   else{
                    out_Jet_pt[j] = 0;
                    out_Jet_eta[j] = 0;
                    out_Jet_phi[j] = 0;
                    out_Jet_btagDeepFlavB[j] = 0;                
                }
            }
            outTree->Fill();
        }
    }

    outFile->Write();
    outFile->Close();
}

int main() {
    double wsum = 0.0;
    TChain* tree = createChainFromFileList("../data/real/CMS_Run2016H_DoubleEG_NANOAOD_UL2016_MiniAODv2_NanoAODv9-v1_100000_file_index.txt", 1);  
    TChain* mcTreeHigh = createChainFromFileList("../data/raw/signal/CMS_mc_RunIISummer20UL16NanoAODv9_DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_NANOAODSIM_106X_mcRun2_asymptotic_v17-v1_30000_file_index.txt", 1);  
    TChain* mcTreeLow = createChainFromFileList("../data/raw/signal/CMS_mc_RunIISummer20UL16NanoAODv9_DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8_NANOAODSIM_106X_mcRun2_asymptotic_v17-v1_2520000_file_index.txt", 1);

    writeRootFile(tree, "../data/processed/real/real.root", wsum, false);
    writeRootFile(mcTreeHigh, "../data/processed/signal/mcDYhigh.root", wsum, true);
    cout << "Wsum for mcTreeHigh: " << wsum << endl;
    writeRootFile(mcTreeLow, "../data/processed/signal/mcDYlow.root", wsum, true);
    cout << "Wsum for mcTreeLow: " << wsum << endl;

    return 0;
}