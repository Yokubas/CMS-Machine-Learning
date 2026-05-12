#include <cstdio> 
#include <cstdlib>
#include <iostream>
#include <memory> 
#include <cmath> 
#include <vector>
#include <fstream>
#include <string> 

#include "TFile.h"     
#include "TTree.h"
#include "TH1D.h"
#include "TCanvas.h"
#include "Math/Vector4D.h"
#include "TF1.h"
#include "TChain.h"
#include "TLegend.h"
#include "TStyle.h"
#include "TLine.h"
#include "THStack.h"
#include "TH2.h"

using ROOT::Math::PtEtaPhiMVector;

using namespace std;

// const double electron_mass = 0.000511; // GeV
const float isoCut = 0.15;

void fillHistogramFromTree(TTree* tree, TH1D& hist, TH1D& hist_tautau, bool isMC, double& sumGenWeight, bool isDY){
    
    tree->SetBranchStatus("*", false);
    tree->SetBranchStatus("Electron_charge", true);
    tree->SetBranchStatus("Electron_cutBased", true);
    tree->SetBranchStatus("Electron_pfRelIso03_all", true);
    tree->SetBranchStatus("nElectron", true);
    tree->SetBranchStatus("Electron_phi", true);
    tree->SetBranchStatus("Electron_eta", true);
    tree->SetBranchStatus("Electron_pt", true);
    tree->SetBranchStatus("Electron_mass", true);
    tree->SetBranchStatus("HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ", true);

    Int_t Electron_charge[100];
    UInt_t nElectron;
    Float_t Electron_phi[100], Electron_eta[100], Electron_pt[100], Electron_pfRelIso03_all[100], Electron_mass[100];
    Int_t Electron_cutBased[100];
    Float_t genWeight = 1.0;
    Bool_t HLT_Ele23_Ele12;

    sumGenWeight = 0;

    if (isMC) {
        tree->SetBranchStatus("genWeight", true);
        tree->SetBranchAddress("genWeight", &genWeight);    
    }

    UInt_t nLHEPart = 0;
    Int_t LHEPart_pdgId[100];

    if (isDY) {
        tree->SetBranchStatus("nLHEPart", true);
        tree->SetBranchStatus("LHEPart_pdgId", true);
        tree->SetBranchAddress("nLHEPart", &nLHEPart);
        tree->SetBranchAddress("LHEPart_pdgId", &LHEPart_pdgId);
    }

    tree->SetBranchAddress("Electron_charge", &Electron_charge);
    tree->SetBranchAddress("nElectron", &nElectron);
    tree->SetBranchAddress("Electron_mass", &Electron_mass);
    tree->SetBranchAddress("Electron_pt", &Electron_pt);
    tree->SetBranchAddress("Electron_eta", &Electron_eta);
    tree->SetBranchAddress("Electron_phi", &Electron_phi);
    tree->SetBranchAddress("Electron_cutBased", &Electron_cutBased);
    tree->SetBranchAddress("Electron_pfRelIso03_all", &Electron_pfRelIso03_all);
    tree->SetBranchAddress("HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ", &HLT_Ele23_Ele12);

    for (int iEntry = 0; tree->LoadTree(iEntry) >= 0; ++iEntry){    
        
        tree->GetEntry(iEntry);

        if (isMC) sumGenWeight += genWeight;
    
        if (!HLT_Ele23_Ele12) continue;

        vector<int> goodIndices;

        for (UInt_t i = 0; i < nElectron; ++i){
            Bool_t isGoodElectron = (Electron_cutBased[i] >= 3) && (Electron_pfRelIso03_all[i] < isoCut);
            if (isGoodElectron) {
                goodIndices.push_back(i);
            }
        }

        if (goodIndices.size() > 1){

            sort(goodIndices.begin(), goodIndices.end(), [&](int a, int b) {
                return Electron_pt[a] > Electron_pt[b];
            });
            
            int leading = goodIndices[0];
            int subleading = goodIndices[1];

            if (Electron_charge[leading] * Electron_charge[subleading] < 0) {

                if (Electron_pt[leading] > 28 && Electron_pt[subleading] > 20 && 
                    fabs(Electron_eta[leading]) < 2.5 && fabs(Electron_eta[subleading]) < 2.5 && 
                    !(fabs(Electron_eta[leading]) > 1.4442 && fabs(Electron_eta[leading]) < 1.566) &&
                    !(fabs(Electron_eta[subleading]) > 1.4442 && fabs(Electron_eta[subleading]) < 1.566)) {

                    PtEtaPhiMVector electron1(Electron_pt[leading], Electron_eta[leading], Electron_phi[leading], Electron_mass[leading]);
                    PtEtaPhiMVector electron2(Electron_pt[subleading], Electron_eta[subleading], Electron_phi[subleading], Electron_mass[subleading]);

                    auto dilepton = electron1 + electron2;
                    
                    bool hasTau = false;
                    
                    if (isDY) {
                        for (UInt_t i = 0; i < nLHEPart; ++i) {
                            if (abs(LHEPart_pdgId[i]) == 15) {
                                hasTau = true;
                                break;  
                            }
                        }
                    }

                    double weight = genWeight;

                    if (hasTau) hist_tautau.Fill(dilepton.M(), weight);
                    
                    else hist.Fill(dilepton.M(), weight);
                }
            }    
        }
    }
}

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
    double Wsum = 0.0;

    double sigmaDY = 6422.0; 
    double sigmaDYLow = 20480.0;
    double sigmaTTbar = 756.1;

    double sigmaTW = 32.45;     
    double sigmaAW = 32.51;     
    double sigmaST = 119.7;     
    double sigmaSA = 71.74;

    double sigmaZZ = 12.08;     
    double sigmaWZ = 27.56;     
    double sigmaWW = 75.88;
    
    double L_int = 8746231868.215154648 / 1e6; // pb
    
    int nTotalData = 85388673;

    double bins[] ={40,45,50,55,60,64,68,72,76,81,86,91,96,101,106,110,115,120,126,133,141,150,160,171,185,200,220,243,273,320,380,440,510,600,700,830,1000,1500,2000,3000};
    int nbins = sizeof(bins)/sizeof(double) - 1;
    
    TChain* tree = createChainFromFileList("../data/real/CMS_Run2016H_DoubleEG_NANOAOD_UL2016_MiniAODv2_NanoAODv9-v1_100000_file_index.txt", 1);  
    TChain* mcTreeHigh = createChainFromFileList("../data/raw/plot/signal/CMS_mc_RunIISummer20UL16NanoAODv9_DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_NANOAODSIM_106X_mcRun2_asymptotic_v17-v1_30000_file_index.txt", 1);  
    TChain* mcTreeLow = createChainFromFileList("../data/raw/plot/signal/CMS_mc_RunIISummer20UL16NanoAODv9_DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8_NANOAODSIM_106X_mcRun2_asymptotic_v17-v1_2520000_file_index.txt", 1);
    // Background
    TChain* ttbarTree = createChainFromFileList("../data/raw/plot/background/CMS_mc_RunIISummer20UL16NanoAODv9_TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8_NANOAODSIM_106X_mcRun2_asymptotic_v17-v1_40000_file_index.txt", 4);

    TChain* twTree = createChainFromFileList("../data/raw/plot/background/CMS_mc_RunIISummer20UL16NanoAODv9_ST_tW_top_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8_NANOAODSIM_106X_mcRun2_asymptotic_v17-v1_2520000_file_index.txt", 4);  
    TChain* awTree = createChainFromFileList("../data/raw/plot/background/CMS_mc_RunIISummer20UL16NanoAODv9_ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8_NANOAODSIM_106X_mcRun2_asymptotic_v17-v1_270000_file_index.txt", 4);  
    TChain* stTree = createChainFromFileList("../data/raw/plot/background/CMS_mc_RunIISummer20UL16NanoAODv9_ST_t-channel_top_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8_NANOAODSIM_106X_mcRun2_asymptotic_v17-v1_130000_file_index.txt", 4);  
    TChain* saTree = createChainFromFileList("../data/raw/plot/background/CMS_mc_RunIISummer20UL16NanoAODv9_ST_t-channel_antitop_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8_NANOAODSIM_106X_mcRun2_asymptotic_v17-v1_130000_file_index.txt");  
   
    TChain* zzTree = createChainFromFileList("../data/raw/plot/background/CMS_mc_RunIISummer20UL16NanoAODv9_ZZ_TuneCP5_13TeV-pythia8_NANOAODSIM_106X_mcRun2_asymptotic_v17-v1_270000_file_index.txt");  
    TChain* wzTree = createChainFromFileList("../data/raw/plot/background/CMS_mc_RunIISummer20UL16NanoAODv9_WZ_TuneCP5_13TeV-pythia8_NANOAODSIM_106X_mcRun2_asymptotic_v17-v1_270000_file_index.txt");  
    TChain* wwTree = createChainFromFileList("../data/raw/plot/background/CMS_mc_RunIISummer20UL16NanoAODv9_WW_TuneCP5_13TeV-pythia8_NANOAODSIM_106X_mcRun2_asymptotic_v17-v1_270000_file_index.txt", 4);  

    TH1D hDummy("hDummy", "", 1, 0, 1);

    TH1D hData("hData", "Dilepton Invariant Mass", nbins, bins);

    TH1D hDY("hDY", "Drell-Yan MC", nbins, bins);
    TH1D hDY_tau("hDY_tau", "Drell-Yan -> TauTau MC", nbins, bins);

    TH1D hDYLow("hDYLow", "Drell-Yan Low MC", nbins, bins);
    TH1D hDYLow_tau("hDYLow_tau", "Drell-Yan Low -> TauTau MC", nbins, bins);

    // Bakcground
    TH1D hTTbar("hTTbar", "TTbar MC", nbins, bins);

    TH1D hTW("hTW", "TW MC", nbins, bins);
    TH1D hAW("hAW", "AW MC", nbins, bins);
    TH1D hST("hST", "ST MC", nbins, bins);
    TH1D hSA("hSA", "SA MC", nbins, bins);

    TH1D hZZ("hZZ", "ZZ MC", nbins, bins);
    TH1D hWZ("hWZ", "WZ MC", nbins, bins);
    TH1D hWW("hWW", "WW MC", nbins, bins);

    // Real data
    fillHistogramFromTree(tree, hData, hDummy, false, Wsum, false);

    double nEntriesData = tree->GetEntries();

    // DY high with tautau
    fillHistogramFromTree(mcTreeHigh, hDY, hDY_tau, true, Wsum, true);
    
    hDY.Scale((sigmaDY*L_int*(nEntriesData/nTotalData))/Wsum);
    hDY_tau.Scale(sigmaDY*L_int*(nEntriesData/nTotalData)/Wsum);

    // DY low with tautau
    fillHistogramFromTree(mcTreeLow, hDYLow, hDYLow_tau, true, Wsum, true);

    hDYLow.Scale((sigmaDYLow*L_int*(nEntriesData/nTotalData))/Wsum);
    hDYLow_tau.Scale(sigmaDYLow*L_int*(nEntriesData/nTotalData)/Wsum);

    // TTbar
    fillHistogramFromTree(ttbarTree, hTTbar, hDummy, true, Wsum, false);

    hTTbar.Scale(sigmaTTbar*L_int*(nEntriesData/nTotalData)/Wsum);

    // TW
    fillHistogramFromTree(twTree, hTW, hDummy, true, Wsum, false);

    hTW.Scale(sigmaTW*L_int*(nEntriesData/nTotalData)/Wsum);

    // AW
    fillHistogramFromTree(awTree, hAW, hDummy, true, Wsum, false);

    hAW.Scale(sigmaAW*L_int*(nEntriesData/nTotalData)/Wsum);

    // ST
    fillHistogramFromTree(stTree, hST, hDummy, true, Wsum, false);

    hST.Scale((sigmaST*L_int*(nEntriesData/nTotalData))/Wsum);

    // SA
    fillHistogramFromTree(saTree, hSA, hDummy, true, Wsum, false);

    hSA.Scale((sigmaSA*L_int*(nEntriesData/nTotalData))/Wsum);

    // ZZ
    fillHistogramFromTree(zzTree, hZZ, hDummy, true, Wsum, false);

    hZZ.Scale((sigmaZZ*L_int*(nEntriesData/nTotalData))/Wsum);

    // WZ
    fillHistogramFromTree(wzTree, hWZ, hDummy, true, Wsum, false);

    hWZ.Scale((sigmaWZ*L_int*(nEntriesData/nTotalData))/Wsum);

    // WW
    fillHistogramFromTree(wwTree, hWW, hDummy, true, Wsum, false);

    hWW.Scale(sigmaWW*L_int*(nEntriesData/nTotalData)/Wsum);

    TH1D hDYTotal("hDYTotal", "Total DY MC", nbins, bins);
    hDYTotal.Add(&hDY);
    hDYTotal.Add(&hDYLow);

    TH1D hDYTotal_tau("hDYTotal_tau", "Total DY TauTau MC", nbins, bins);
    hDYTotal_tau.Add(&hDY_tau);
    hDYTotal_tau.Add(&hDYLow_tau);

    TH1D hSTotal("hSTotal", "Total Single Top MC", nbins, bins);
    hSTotal.Add(&hTW);
    hSTotal.Add(&hAW);
    hSTotal.Add(&hST);
    hSTotal.Add(&hSA);

    TCanvas* c = new TCanvas("c", "Data vs MC", 800, 800);
   
    c->Divide(1,2);

    c->cd(1);

    gPad->SetTickx(1); 
    gPad->SetTicky(1);
    gPad->SetPad(0, 0.3, 1, 1.0);
    gPad->SetBottomMargin(0.01);
    gPad->SetGrid();
    hData.GetYaxis()->SetTitle("Counts");
    gPad->SetLogy();
    gPad->SetLogx();

    hData.GetXaxis()->SetTitleSize(0);
    hData.GetXaxis()->SetLabelSize(0);      
    hData.GetYaxis()->SetTitleSize(0.04);  
    hData.GetYaxis()->SetLabelSize(0.04); 

    hData.SetMarkerStyle(20);       
    hData.SetMarkerSize(0.75);       
    hData.SetLineColor(kBlack);     
    hData.SetMarkerColor(kBlack);

    THStack* hStack = new THStack("hStack", "Dilepton Invariant Mass");

    hDYTotal.SetFillColor(kRed-7);
    hDYTotal_tau.SetFillColor(kViolet+2);     
    hTTbar.SetFillColor(kBlue-7); 
    hSTotal.SetFillColor(kGreen+2);
    hZZ.SetFillColor(kOrange-3);
    hWZ.SetFillColor(kMagenta-5);
    hWW.SetFillColor(kCyan-6);

    hStack->Add(&hWW);
    hStack->Add(&hWZ);
    hStack->Add(&hZZ);
    hStack->Add(&hSTotal);
    hStack->Add(&hTTbar);
    hStack->Add(&hDYTotal_tau);
    hStack->Add(&hDYTotal);

    gStyle->SetOptStat(0);

    hData.Draw("EP");       
    hStack->Draw("HIST same");
    hData.Draw("EP same");         
    
    TLegend *l = new TLegend(0.64, 0.56, 0.86, 0.86);
    l->AddEntry(&hData, "Data", "ep");
    l->AddEntry(&hDYTotal, "DY #rightarrow ee", "f");
    l->AddEntry(&hDYTotal_tau, "DY #rightarrow #tau#tau", "f");
    l->AddEntry(&hTTbar, "t#bar{t}", "f");
    l->AddEntry(&hSTotal, "Single Top", "f");
    l->AddEntry(&hZZ, "ZZ", "f");
    l->AddEntry(&hWZ, "WZ", "f");
    l->AddEntry(&hWW, "WW", "f");

    l->Draw();

    c->cd(2); 
    gPad->SetGrid();
    gPad->SetPad(0, 0.0, 1, 0.3);  
    gPad->SetBottomMargin(0.35);
    gPad->SetTickx(1);  
    gPad->SetTicky(1);
    gPad->SetLogx();

    TH1D hMCSum("hMCSum", "MC Sum", nbins, bins);
    hMCSum.Add(&hDYTotal);
    hMCSum.Add(&hDYTotal_tau);
    hMCSum.Add(&hTTbar);
    hMCSum.Add(&hSTotal);
    hMCSum.Add(&hZZ);
    hMCSum.Add(&hWZ);
    hMCSum.Add(&hWW);

    TH1D* hRatio = (TH1D*)hData.Clone("hRatio");
    hRatio->Divide(&hMCSum);  

    hRatio->SetTitle("");
    hRatio->SetMinimum(0.5);
    hRatio->SetMaximum(1.5);

    hRatio->GetYaxis()->SetTitle("Data / Pred.");
    hRatio->GetYaxis()->SetNdivisions(505);
    hRatio->GetYaxis()->SetTitleSize(0.12);
    hRatio->GetYaxis()->SetLabelSize(0.10);
    hRatio->GetYaxis()->SetTitleOffset(0.4);

    hRatio->SetMarkerStyle(20);       
    hRatio->SetMarkerSize(0.7);       
    hRatio->SetLineColor(kBlack);     
    hRatio->SetMarkerColor(kBlack);

    hRatio->GetXaxis()->SetTitle("Mass [GeV]");
    hRatio->GetXaxis()->SetTitleSize(0.12);
    hRatio->GetXaxis()->SetLabelSize(0.10);    

    hRatio->Draw("EP");

    TLine* line = new TLine(40, 1, 3000, 1);
    line->SetLineColor(kBlack);
    line->SetLineWidth(1);
    line->Draw("same");

    TH1D hBackground("hBackground", "Background", nbins, bins);
    hBackground.Add(&hDYTotal_tau);
    hBackground.Add(&hTTbar);
    hBackground.Add(&hSTotal);
    hBackground.Add(&hZZ);
    hBackground.Add(&hWZ);
    hBackground.Add(&hWW);

    cout << "\n===== Per-bin S/(S+B) =====\n";

    for (int i = 1; i <= nbins; i++) {

        double S = hDYTotal.GetBinContent(i);
        double B = hBackground.GetBinContent(i);

        double denom = S + B;

        double ratio = (denom > 0) ? (S / denom) : 0.0;

        double massLow  = hDYTotal.GetBinLowEdge(i);
        double massHigh = hDYTotal.GetBinLowEdge(i+1);

        cout << "Bin [" << massLow << ", " << massHigh << "] "
            << "S=" << S << " "
            << "B=" << B << " "
            << "S/(S+B)=" << ratio
            << endl;
    }

    cout << "\n===== DY TOTAL COUNTS PER BIN =====\n";

    for (int i = 1; i <= nbins; i++) {

        double S = hDYTotal.GetBinContent(i);

        double massLow  = hDYTotal.GetBinLowEdge(i);
        double massHigh = hDYTotal.GetBinLowEdge(i+1);

        cout << "Bin [" << massLow << ", " << massHigh << "] "
            << "DY counts = " << S
            << endl;
    }

    c->SaveAs("../results/stack_mediumid.png");

    delete c;

    return 0;
}