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
#include <filesystem>
using namespace std;

Float_t Electron_pt[100], Electron_eta[100], Electron_phi[100], Electron_mass[100];
// Float_t Electron_miniPFRelIso_all[100], Electron_miniPFRelIso_chg[100];
// Float_t Electron_dz[100], Electron_dxy[100], Electron_ip3d[100], Electron_charge[100];

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

void writeRootFile(TTree* tree, const char* outFileName, double& sumGenWeight, bool isMC, bool isDY, bool isWJets){
    
    tree->SetBranchStatus("*", false);
    tree->SetBranchStatus("Electron_pt", true);
    tree->SetBranchStatus("Electron_eta", true);
    tree->SetBranchStatus("Electron_phi", true);
    tree->SetBranchStatus("nElectron", true);
    tree->SetBranchStatus("Electron_mass", true);
    tree->SetBranchStatus("Electron_sieie", true);
    tree->SetBranchStatus("Electron_hoe", true);
    tree->SetBranchStatus("Electron_dz", true);
    tree->SetBranchStatus("MET_phi", true);
    tree->SetBranchStatus("MET_significance", true);
    tree->SetBranchStatus("MET_sumEt", true);
    tree->SetBranchStatus("Electron_scEtOverPt", true);
    tree->SetBranchStatus("Electron_miniPFRelIso_all", true);
    tree->SetBranchStatus("Electron_eInvMinusPInv", true);
    tree->SetBranchStatus("Electron_dxy", true);
    tree->SetBranchStatus("Electron_dr03TkSumPt", true);

    tree->SetBranchStatus("nJet", true);
    tree->SetBranchStatus("Jet_pt", true);
    tree->SetBranchStatus("Jet_eta", true);
    tree->SetBranchStatus("Jet_phi", true);
    tree->SetBranchStatus("Jet_btagDeepFlavB", true);
    
    UChar_t Electron_genPartFlav[100];
    UInt_t nLHEPart = 0;
    Int_t LHEPart_pdgId[100];
    Float_t genWeight;
    Float_t Electron_sieie[100], MET_phi[100], MET_significance[100], MET_sumEt[100];
    Float_t Electron_hoe[100], Electron_scEtOverPt[100], Electron_miniPFRelIso_all[100];
    Float_t Electron_dz[100], Electron_eInvMinusPInv[100], Electron_dxy[100], Electron_dr03TkSumPt[100];

    sumGenWeight = 0.0;
    Float_t out_genWeight;
    UInt_t nElectron;
    UInt_t nJet;
    UInt_t out_nElectron;
    
    Float_t out_Electron_pt[2], out_Electron_eta[2], out_Electron_phi[2], out_Electron_mass[2], out_Electron_eInvMinusPInv[2], out_Electron_dxy[2]; 
    Float_t out_Electron_sieie[2], out_Electron_hoe[2], out_Electron_dz[2], out_Electron_scEtOverPt[2], out_Electron_miniPFRelIso_all[2], out_Electron_dr03TkSumPt[2];
    Float_t out_MET_phi, out_MET_significance, out_MET_sumEt;

    Float_t out_Jet_pt[4], out_Jet_eta[4], out_Jet_phi[4], out_Jet_btagDeepFlavB[4];
    
    if (isMC) {
        tree->SetBranchStatus("genWeight", true);
        tree->SetBranchAddress("genWeight", &genWeight);    
    }
    
    if (isDY) {
        tree->SetBranchStatus("nLHEPart", true);
        tree->SetBranchStatus("LHEPart_pdgId", true);
        tree->SetBranchAddress("nLHEPart", &nLHEPart);
        tree->SetBranchAddress("LHEPart_pdgId", &LHEPart_pdgId);
    }
    
    if(isWJets){
        tree->SetBranchStatus("Electron_genPartFlav", true);
        tree->SetBranchAddress("Electron_genPartFlav", &Electron_genPartFlav);
    }
    
    tree->SetBranchAddress("Electron_pt", &Electron_pt);
    tree->SetBranchAddress("Electron_eta", &Electron_eta);
    tree->SetBranchAddress("Electron_phi", &Electron_phi);
    tree->SetBranchAddress("nElectron", &nElectron);
    tree->SetBranchAddress("Electron_mass", &Electron_mass);
    tree->SetBranchAddress("Electron_sieie", &Electron_sieie);
    tree->SetBranchAddress("Electron_hoe", &Electron_hoe);
    tree->SetBranchAddress("Electron_dz", &Electron_dz);
    tree->SetBranchAddress("MET_phi", &MET_phi);
    tree->SetBranchAddress("MET_significance", &MET_significance);
    tree->SetBranchAddress("MET_sumEt", &MET_sumEt);
    tree->SetBranchAddress("Electron_scEtOverPt", &Electron_scEtOverPt);
    tree->SetBranchAddress("Electron_miniPFRelIso_all", &Electron_miniPFRelIso_all);
    tree->SetBranchAddress("Electron_eInvMinusPInv", &Electron_eInvMinusPInv);
    tree->SetBranchAddress("Electron_dxy", &Electron_dxy);
    tree->SetBranchAddress("Electron_dr03TkSumPt", &Electron_dr03TkSumPt);

    tree->SetBranchAddress("nJet", &nJet);
    tree->SetBranchAddress("Jet_pt", &Jet_pt);
    tree->SetBranchAddress("Jet_eta", &Jet_eta);
    tree->SetBranchAddress("Jet_phi", &Jet_phi);
    tree->SetBranchAddress("Jet_btagDeepFlavB", &Jet_btagDeepFlavB);
    
    TFile *outFile = new TFile(outFileName,"RECREATE");
    TTree *outTree = new TTree("Events","Selected branches");

    TFile *outFile_tau = nullptr;
    TTree *outTree_tau = nullptr;

    if (isDY) {
        string outNameStr(outFileName); // "../data/processed/signal/mcDYlow.root"
        std::filesystem::path p(outNameStr);

        string baseName = p.stem().string();     // "mcDYlow"
        string tauFilePath = "../data/processed/background/" + baseName + "_tau.root";

        outFile_tau = new TFile(tauFilePath.c_str(), "RECREATE");
        outTree_tau = new TTree("Events","DY tau");
        outTree_tau->Branch("nElectron", &out_nElectron, "nElectron/i");
        outTree_tau->Branch("Electron_pt", &out_Electron_pt, "Electron_pt[2]/F");
        outTree_tau->Branch("Electron_eta", &out_Electron_eta, "Electron_eta[2]/F");
        outTree_tau->Branch("Electron_phi", &out_Electron_phi, "Electron_phi[2]/F");
        outTree_tau->Branch("Electron_mass", &out_Electron_mass, "Electron_mass[2]/F");
        outTree_tau->Branch("Electron_sieie", &out_Electron_sieie, "Electron_sieie[2]/F");
        outTree_tau->Branch("Electron_hoe", &out_Electron_hoe, "Electron_hoe[2]/F");
        outTree_tau->Branch("Electron_dz", &out_Electron_dz, "Electron_dz[2]/F");
        outTree_tau->Branch("MET_phi", &out_MET_phi, "MET_phi/F");
        outTree_tau->Branch("MET_significance", &out_MET_significance, "MET_significance/F");
        outTree_tau->Branch("MET_sumEt", &out_MET_sumEt, "MET_sumEt/F");
        outTree_tau->Branch("Electron_scEtOverPt", &out_Electron_scEtOverPt, "Electron_scEtOverPt[2]/F");
        outTree_tau->Branch("Electron_miniPFRelIso_all", &out_Electron_miniPFRelIso_all, "Electron_miniPFRelIso_all[2]/F");
        outTree_tau->Branch("Electron_eInvMinusPInv", &out_Electron_eInvMinusPInv, "Electron_eInvMinusPInv[2]/F");
        outTree_tau->Branch("Electron_dxy", &out_Electron_dxy, "Electron_dxy[2]/F");
        outTree_tau->Branch("Electron_dr03TkSumPt", &out_Electron_dr03TkSumPt, "Electron_dr03TkSumPt[2]/F");

        if(isMC) outTree_tau->Branch("genWeight", &out_genWeight, "genWeight/F");

        outTree_tau->Branch("Jet_pt", &out_Jet_pt, "Jet_pt[4]/F");
        outTree_tau->Branch("Jet_eta", &out_Jet_eta, "Jet_eta[4]/F");
        outTree_tau->Branch("Jet_phi", &out_Jet_phi, "Jet_phi[4]/F");
        outTree_tau->Branch("Jet_btagDeepFlavB", &out_Jet_btagDeepFlavB, "Jet_btagDeepFlavB[4]/F");
    }

    // output branches
    outTree->Branch("nElectron", &out_nElectron, "nElectron/i");
    outTree->Branch("Electron_pt", &out_Electron_pt, "Electron_pt[2]/F");
    outTree->Branch("Electron_eta", &out_Electron_eta, "Electron_eta[2]/F");
    outTree->Branch("Electron_phi", &out_Electron_phi, "Electron_phi[2]/F");
    outTree->Branch("Electron_mass", &out_Electron_mass, "Electron_mass[2]/F");
    outTree->Branch("Electron_sieie", &out_Electron_sieie, "Electron_sieie[2]/F");
    outTree->Branch("Electron_hoe", &out_Electron_hoe, "Electron_hoe[2]/F");
    outTree->Branch("Electron_dz", &out_Electron_dz, "Electron_dz[2]/F");
    outTree->Branch("MET_phi", &out_MET_phi, "MET_phi/F");
    outTree->Branch("MET_significance", &out_MET_significance, "MET_significance/F");
    outTree->Branch("MET_sumEt", &out_MET_sumEt, "MET_sumEt/F");
    outTree->Branch("Electron_scEtOverPt", &out_Electron_scEtOverPt, "Electron_scEtOverPt[2]/F");
    outTree->Branch("Electron_miniPFRelIso_all", &out_Electron_miniPFRelIso_all, "Electron_miniPFRelIso_all[2]/F");
    outTree->Branch("Electron_eInvMinusPInv", &out_Electron_eInvMinusPInv, "Electron_eInvMinusPInv[2]/F");
    outTree->Branch("Electron_dxy", &out_Electron_dxy, "Electron_dxy[2]/F");
    outTree->Branch("Electron_dr03TkSumPt", &out_Electron_dr03TkSumPt, "Electron_dr03TkSumPt[2]/F");

    if(isMC) outTree->Branch("genWeight", &out_genWeight, "genWeight/F");


    outTree->Branch("Jet_pt", out_Jet_pt, "Jet_pt[4]/F");
    outTree->Branch("Jet_eta", out_Jet_eta, "Jet_eta[4]/F");
    outTree->Branch("Jet_phi", out_Jet_phi, "Jet_phi[4]/F");
    outTree->Branch("Jet_btagDeepFlavB", out_Jet_btagDeepFlavB, "Jet_btagDeepFlavB[4]/F");

    for (int iEntry = 0; tree->LoadTree(iEntry) >= 0; ++iEntry){    
        
        tree->GetEntry(iEntry);
        if (isWJets) {
            vector<int> wjetsElectrons;

            for (UInt_t i = 0; i < nElectron; ++i) {
                if (Electron_genPartFlav[i] != 1) {
                    wjetsElectrons.push_back(i);  // keep this "fake"/background electron
                }
            }
            if(wjetsElectrons.size() < 2) continue;

            sort(wjetsElectrons.begin(), wjetsElectrons.end(), [&](int a, int b) {
                return Electron_pt[a] > Electron_pt[b];});
                
            int idx1 = wjetsElectrons[0];  // leading
            int idx2 = wjetsElectrons[1];  // subleading

            // copy values to output variables
            out_nElectron = 2;
            
            int electronIndices[2] = {idx1, idx2};
            
            if (Electron_pt[idx1] > 28 && Electron_pt[idx2] > 20 && 
                    fabs(Electron_eta[idx1]) < 2.5 && fabs(Electron_eta[idx2]) < 2.5 && 
                    !(fabs(Electron_eta[idx1]) > 1.4442 && fabs(Electron_eta[idx1]) < 1.566) &&
                    !(fabs(Electron_eta[idx2]) > 1.4442 && fabs(Electron_eta[idx2]) < 1.566)) {

                        for (int i = 0; i < 2; i++) {
            
                            int id = electronIndices[i];
            
                            out_Electron_pt[i]  = Electron_pt[id];
                            out_Electron_eta[i] = Electron_eta[id];
                            out_Electron_phi[i] = Electron_phi[id];
                            out_Electron_mass[i] = Electron_mass[id];
                            out_Electron_dz[i] = Electron_dz[id];
                            out_Electron_hoe[i] = Electron_hoe[id];
                            out_Electron_sieie[i] = Electron_sieie[id];
                            out_Electron_scEtOverPt[i] = Electron_scEtOverPt[id];
                            out_Electron_miniPFRelIso_all[i] = Electron_miniPFRelIso_all[id];
                            out_Electron_eInvMinusPInv[i] = Electron_eInvMinusPInv[id];
                            out_Electron_dxy[i] = Electron_dxy[id];
                            out_Electron_dr03TkSumPt[i] = Electron_dr03TkSumPt[id];
                        
                        }
                        
                        if (isMC){ 
                            out_genWeight = genWeight;
                        }

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
                
                    }
            out_MET_phi = MET_phi[0];
            out_MET_significance = MET_significance[0];
            out_MET_sumEt = MET_sumEt[0];
            outTree->Fill(); 
            continue; // skip rest of selection for W+jets events
        }

        bool hasTau = false;
        
        if (isDY) {
            for (UInt_t i = 0; i < nLHEPart; ++i) {
                if (abs(LHEPart_pdgId[i]) == 15) {
                    hasTau = true;
                    break;
                }
            }
        }
        
        if (isMC) {
                sumGenWeight += genWeight; 
        }
        vector<int> twoElectrons;

        if (nElectron > 1){
            
            for (UInt_t i = 0; i < nElectron; ++i){
                twoElectrons.push_back(i);
            }
            sort(twoElectrons.begin(), twoElectrons.end(), [&](int a, int b) {
                return Electron_pt[a] > Electron_pt[b];});
                
            int idx1 = twoElectrons[0];  // leading
            int idx2 = twoElectrons[1];  // subleading

            // copy values to output variables
            out_nElectron = 2;
            
            int electronIndices[2] = {idx1, idx2};
            
            if (Electron_pt[idx1] > 28 && Electron_pt[idx2] > 20 && 
                    fabs(Electron_eta[idx1]) < 2.5 && fabs(Electron_eta[idx2]) < 2.5 && 
                    !(fabs(Electron_eta[idx1]) > 1.4442 && fabs(Electron_eta[idx1]) < 1.566) &&
                    !(fabs(Electron_eta[idx2]) > 1.4442 && fabs(Electron_eta[idx2]) < 1.566)) {

                        for (int i = 0; i < 2; i++) {
            
                            int id = electronIndices[i];
            
                            out_Electron_pt[i]  = Electron_pt[id];
                            out_Electron_eta[i] = Electron_eta[id];
                            out_Electron_phi[i] = Electron_phi[id];
                            out_Electron_mass[i] = Electron_mass[id];
                            out_Electron_dz[i] = Electron_dz[id];
                            out_Electron_hoe[i] = Electron_hoe[id];
                            out_Electron_sieie[i] = Electron_sieie[id];
                            out_Electron_scEtOverPt[i] = Electron_scEtOverPt[id];
                            out_Electron_miniPFRelIso_all[i] = Electron_miniPFRelIso_all[id];
                            out_Electron_eInvMinusPInv[i] = Electron_eInvMinusPInv[id];
                            out_Electron_dxy[i] = Electron_dxy[id];
                            out_Electron_dr03TkSumPt[i] = Electron_dr03TkSumPt[id];
                        }
                        
                        if (isMC){ 
                            out_genWeight = genWeight;
                        }

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
                        
                        if (isDY) {
                            if (hasTau) {
                                out_MET_phi = MET_phi[0];
                                out_MET_significance = MET_significance[0];
                                out_MET_sumEt = MET_sumEt[0];
                                outTree_tau->Fill();      // goes to tau file
                            } else {
                                out_MET_phi = MET_phi[0];
                                out_MET_significance = MET_significance[0];
                                out_MET_sumEt = MET_sumEt[0];
                                outTree->Fill();   // clean DY only
                            }
                        } else {
                            out_MET_phi = MET_phi[0];
                            out_MET_significance = MET_significance[0];
                            out_MET_sumEt = MET_sumEt[0];
                            outTree->Fill();       // all non-DY unchanged
                        }
                    }
        }
        
    }

    outFile->Write();
    outFile->Close();
    if (isDY) {
    outFile_tau->Write();
    outFile_tau->Close();
    }
}

int main() {
    double wsum = 0.0;
    TChain* tree = createChainFromFileList("../data/real/CMS_Run2016H_DoubleEG_NANOAOD_UL2016_MiniAODv2_NanoAODv9-v1_100000_file_index.txt", 1);  
    TChain* mcTreeHigh = createChainFromFileList("../data/raw/signal/CMS_mc_RunIISummer20UL16NanoAODv9_DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_NANOAODSIM_106X_mcRun2_asymptotic_v17-v1_30000_file_index.txt", 1);  
    TChain* mcTreeLow = createChainFromFileList("../data/raw/signal/CMS_mc_RunIISummer20UL16NanoAODv9_DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8_NANOAODSIM_106X_mcRun2_asymptotic_v17-v1_2520000_file_index.txt", 1);
    // Background
    TChain* ttbarTree = createChainFromFileList("../data/raw/background/CMS_mc_RunIISummer20UL16NanoAODv9_TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8_NANOAODSIM_106X_mcRun2_asymptotic_v17-v1_2530000_file_index.txt", 4);

    TChain* twTree = createChainFromFileList("../data/raw/background/CMS_mc_RunIISummer20UL16NanoAODv9_ST_tW_top_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8_NANOAODSIM_106X_mcRun2_asymptotic_v17-v1_2520000_file_index.txt", 4);  
    TChain* awTree = createChainFromFileList("../data/raw/background/CMS_mc_RunIISummer20UL16NanoAODv9_ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8_NANOAODSIM_106X_mcRun2_asymptotic_v17-v1_270000_file_index.txt", 4);  
    TChain* stTree = createChainFromFileList("../data/raw/background/CMS_mc_RunIISummer20UL16NanoAODv9_ST_t-channel_top_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8_NANOAODSIM_106X_mcRun2_asymptotic_v17-v1_120000_file_index.txt", 4);  
    TChain* saTree = createChainFromFileList("../data/raw/background/CMS_mc_RunIISummer20UL16NanoAODv9_ST_t-channel_antitop_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8_NANOAODSIM_106X_mcRun2_asymptotic_v17-v1_120000_file_index.txt");  
   
    TChain* zzTree = createChainFromFileList("../data/raw/background/CMS_mc_RunIISummer20UL16NanoAODv9_ZZ_TuneCP5_13TeV-pythia8_NANOAODSIM_106X_mcRun2_asymptotic_v17-v1_130000_file_index.txt");  
    TChain* wzTree = createChainFromFileList("../data/raw/background/CMS_mc_RunIISummer20UL16NanoAODv9_WZ_TuneCP5_13TeV-pythia8_NANOAODSIM_106X_mcRun2_asymptotic_v17-v1_110000_file_index.txt", 4);  
    TChain* wwTree = createChainFromFileList("../data/raw/background/CMS_mc_RunIISummer20UL16NanoAODv9_WW_TuneCP5_13TeV-pythia8_NANOAODSIM_106X_mcRun2_asymptotic_v17-v1_130000_file_index.txt", 4);  

    TChain* wjetsTree = createChainFromFileList("../data/raw/background/CMS_mc_RunIISummer20UL16NanoAODv9_WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_NANOAODSIM_106X_mcRun2_asymptotic_v17-v2_2520000_file_index.txt");  

    writeRootFile(tree, "../data/processed/real/real.root", wsum, false, false, false);

    writeRootFile(mcTreeHigh, "../data/processed/signal/mcDYhigh.root", wsum, true, true, false);
    cout << "Wsum for mcTreeHigh: " << wsum << endl;

    writeRootFile(mcTreeLow, "../data/processed/signal/mcDYlow.root", wsum, true, true, false);
    cout << "Wsum for mcTreeLow: " << wsum << endl;

    writeRootFile(ttbarTree, "../data/processed/background/ttbar.root", wsum, true, false, false);
    cout << "Wsum for top-antitop (ttbar): " << wsum << endl;

    writeRootFile(twTree, "../data/processed/background/tW.root", wsum, true, false, false);
    cout << "Wsum for tW: " << wsum << endl;

    writeRootFile(awTree, "../data/processed/background/antitopW.root", wsum, true, false, false);
    cout << "Wsum for antitop W (aw): " << wsum << endl;

    writeRootFile(stTree, "../data/processed/background/singletop.root", wsum, true, false, false);
    cout << "Wsum for single top (t-channel) (st): " << wsum << endl;

    writeRootFile(saTree, "../data/processed/background/sa.root", wsum, true, false, false);
    cout << "Wsum for single antitop (t-channel) (sa): " << wsum << endl;

    writeRootFile(zzTree, "../data/processed/background/zz.root", wsum, true, false, false);
    cout << "Wsum for ZZ: " << wsum << endl;

    writeRootFile(wzTree, "../data/processed/background/wz.root", wsum, true, false, false);
    cout << "Wsum for WZ: " << wsum << endl;

    writeRootFile(wwTree, "../data/processed/background/ww.root", wsum, true, false, false);
    cout << "Wsum for WW: " << wsum << endl;

    writeRootFile(wjetsTree, "../data/processed/background/wjets.root", wsum, true, false, true);

    return 0;
}