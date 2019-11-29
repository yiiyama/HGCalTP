#include "TTree.h"
#include "TFile.h"
#include "TVector2.h"

#include <vector>
#include <array>
#include <memory>
#include <cmath>
#include <unordered_map>
#include <iostream>
#include <algorithm>

template<class V>
class InputPtr {
public:
  InputPtr() : up_(std::unique_ptr<V>(new V())), addr_(up_.get()) {}
  V** addr() { return &addr_; }
  V& operator*() { return *up_; }
  V* operator->() { return up_.get(); }

private:
  std::unique_ptr<V> up_;
  V* addr_;
};

void
extractNtuples(TTree* _input, char const* _outputFileName, double _minPt = 0., long _nEntries = -1)
{
  auto* outputFile(TFile::Open(_outputFileName, "recreate"));
  auto* output(new TTree("clusters", "HGC TP clusters"));

  constexpr unsigned nBinsThetaPhi(5);
  constexpr unsigned nBinsZ(38);
  constexpr unsigned nBins(nBinsThetaPhi * nBinsThetaPhi * nBinsZ);
  constexpr unsigned maxHitsPerBin(3);
  constexpr unsigned long maxHitsPerCluster(250);

  enum Category {
    kElectron,
    kMuon,
    kPhoton,
    kPi0,
    kNeutralHadron,
    kChargedHadron,
    nCategories
  };

  enum ReducedCategory {
    lEGamma,
    lMuon,
    lPi0,
    lHadron,
    nReducedCategories
  };
  
  TString categories[nCategories] = {
    "electron",
    "muon",
    "photon",
    "pi0",
    "neutral",
    "charged"
  };

  TString reducedCategories[nReducedCategories] = {
    "egamma",
    "muon",
    "pi0",
    "hadron"
  };

  // one-hot labels
  int truth_labels[nCategories];
  int truth_labels_reduced[nReducedCategories];

  float cluster_pt;
  float cluster_eta;
  float cluster_phi;
  float cluster_energy;
  float cluster_x;
  float cluster_y;
  int cluster_zside;
  int cluster_showerlength;
  int cluster_coreshowerlength;
  int cluster_firstlayer;
  int cluster_lastlayer;
  float cluster_seetot;
  float cluster_seemax;
  float cluster_spptot;
  float cluster_sppmax;
  float cluster_szz;
  float cluster_srrtot;
  float cluster_srrmax;
  float cluster_srrmean;
  float cluster_emaxe;
  int cluster_quality;
  float cluster_truth_pt;
  float cluster_truth_eta;
  float cluster_truth_phi;
  float cluster_truth_energy;
  // arrays for binned CNN - size 5 x 5 x 38 x 3
  uint32_t bin_id[maxHitsPerBin][nBins]; // 0
  float bin_eta[maxHitsPerBin][nBins]; // 1
  float bin_theta[maxHitsPerBin][nBins]; // 2
  float bin_phi[maxHitsPerBin][nBins]; // 3
  float bin_x[maxHitsPerBin][nBins]; // 4
  float bin_y[maxHitsPerBin][nBins]; // 5
  float bin_eta_global[maxHitsPerBin][nBins]; // 6
  float bin_theta_global[maxHitsPerBin][nBins]; // 7
  float bin_phi_global[maxHitsPerBin][nBins]; // 8
  float bin_dist_global[maxHitsPerBin][nBins]; // 9
  float bin_x_global[maxHitsPerBin][nBins]; // 10
  float bin_y_global[maxHitsPerBin][nBins]; // 11
  float bin_z_global[maxHitsPerBin][nBins]; // 12
  float bin_energy[maxHitsPerBin][nBins]; // 13
  int bin_layer[maxHitsPerBin][nBins]; // 14
  int bin_wafer[maxHitsPerBin][nBins]; // 15
  int bin_wafertype[maxHitsPerBin][nBins]; // 16
  // arrays for graph NN - maximum 250 hits
  int n_cell;
  uint32_t cell_id[maxHitsPerCluster];
  int cell_layer[maxHitsPerCluster];
  float cell_x[maxHitsPerCluster];
  float cell_y[maxHitsPerCluster];
  float cell_z[maxHitsPerCluster];
  float cell_r[maxHitsPerCluster];
  float cell_eta[maxHitsPerCluster];
  float cell_theta[maxHitsPerCluster];
  float cell_phi[maxHitsPerCluster];
  float cell_dist[maxHitsPerCluster];
  float cell_energy[maxHitsPerCluster];
  int cell_wafer[maxHitsPerCluster];
  int cell_wafertype[maxHitsPerCluster];
  int cell_primary[maxHitsPerCluster];

  for (unsigned iC(0); iC != nCategories; ++iC)
    output->Branch(categories[iC], &(truth_labels[iC]), categories[iC] + "/I");
  for (unsigned iC(0); iC != nReducedCategories; ++iC)
    output->Branch(reducedCategories[iC], &(truth_labels[iC]), categories[iC] + "/I");

  output->Branch("cluster_pt", &cluster_pt, "cluster_pt/F");
  output->Branch("cluster_eta", &cluster_eta, "cluster_eta/F");
  output->Branch("cluster_phi", &cluster_phi, "cluster_phi/F");
  output->Branch("cluster_energy", &cluster_energy, "cluster_energy/F");
  output->Branch("cluster_x", &cluster_x, "cluster_x/F");
  output->Branch("cluster_y", &cluster_y, "cluster_y/F");
  output->Branch("cluster_zside", &cluster_zside, "cluster_zside/I");
  output->Branch("cluster_showerlength", &cluster_showerlength, "cluster_showerlength/I");
  output->Branch("cluster_coreshowerlength", &cluster_coreshowerlength, "cluster_coreshowerlength/I");
  output->Branch("cluster_firstlayer", &cluster_firstlayer, "cluster_firstlayer/I");
  output->Branch("cluster_lastlayer", &cluster_lastlayer, "cluster_lastlayer/I");
  output->Branch("cluster_seetot", &cluster_seetot, "cluster_seetot/F");
  output->Branch("cluster_seemax", &cluster_seemax, "cluster_seemax/F");
  output->Branch("cluster_spptot", &cluster_spptot, "cluster_spptot/F");
  output->Branch("cluster_sppmax", &cluster_sppmax, "cluster_sppmax/F");
  output->Branch("cluster_szz", &cluster_szz, "cluster_szz/F");
  output->Branch("cluster_srrtot", &cluster_srrtot, "cluster_srrtot/F");
  output->Branch("cluster_srrmax", &cluster_srrmax, "cluster_srrmax/F");
  output->Branch("cluster_srrmean", &cluster_srrmean, "cluster_srrmean/F");
  output->Branch("cluster_emaxe", &cluster_emaxe, "cluster_emaxe/F");
  output->Branch("cluster_quality", &cluster_quality, "cluster_quality/I");
  output->Branch("cluster_truth_pt", &cluster_truth_pt, "cluster_truth_pt/F");
  output->Branch("cluster_truth_eta", &cluster_truth_eta, "cluster_truth_eta/F");
  output->Branch("cluster_truth_phi", &cluster_truth_phi, "cluster_truth_phi/F");
  output->Branch("cluster_truth_energy", &cluster_truth_energy, "cluster_truth_energy/F");

  auto arrdef(TString::Format("[%d]", nBins));
  for (unsigned iHit(0); iHit != maxHitsPerBin; ++iHit) {
    auto suffix(TString::Format("_%d", iHit));
    output->Branch("bin_id" + suffix, bin_id[iHit], "bin_id" + suffix + arrdef + "/i");
    output->Branch("bin_eta" + suffix, bin_eta[iHit], "bin_eta" + suffix + arrdef + "/F");
    output->Branch("bin_theta" + suffix, bin_theta[iHit], "bin_theta" + suffix + arrdef + "/F");
    output->Branch("bin_phi" + suffix, bin_phi[iHit], "bin_phi" + suffix + arrdef + "/F");
    output->Branch("bin_x" + suffix, bin_x[iHit], "bin_x" + suffix + arrdef + "/F");
    output->Branch("bin_y" + suffix, bin_y[iHit], "bin_y" + suffix + arrdef + "/F");
    output->Branch("bin_eta_global" + suffix, bin_eta_global[iHit], "bin_eta_global" + suffix + arrdef + "/F");
    output->Branch("bin_theta_global" + suffix, bin_theta_global[iHit], "bin_theta_global" + suffix + arrdef + "/F");
    output->Branch("bin_phi_global" + suffix, bin_phi_global[iHit], "bin_phi_global" + suffix + arrdef + "/F");
    output->Branch("bin_dist_global" + suffix, bin_dist_global[iHit], "bin_dist_global" + suffix + arrdef + "/F");
    output->Branch("bin_x_global" + suffix, bin_x_global[iHit], "bin_x_global" + suffix + arrdef + "/F");
    output->Branch("bin_y_global" + suffix, bin_y_global[iHit], "bin_y_global" + suffix + arrdef + "/F");
    output->Branch("bin_z_global" + suffix, bin_z_global[iHit], "bin_z_global" + suffix + arrdef + "/F");
    output->Branch("bin_energy" + suffix, bin_energy[iHit], "bin_energy" + suffix + arrdef + "/F");
    output->Branch("bin_layer" + suffix, bin_layer[iHit], "bin_layer" + suffix + arrdef + "/I");
    output->Branch("bin_wafer" + suffix, bin_wafer[iHit], "bin_wafer" + suffix + arrdef + "/I");
    output->Branch("bin_wafertype" + suffix, bin_wafertype[iHit], "bin_wafertype" + suffix + arrdef + "/I");
  }

  arrdef = TString::Format("[%lu]", maxHitsPerCluster);

  output->Branch("n_cell", &n_cell, "n_cell/I");
  output->Branch("cell_id", cell_id, "cell_id" + arrdef + "/i");
  output->Branch("cell_layer", cell_layer, "cell_layer" + arrdef + "/I");
  output->Branch("cell_x", cell_x, "cell_x" + arrdef + "/F");
  output->Branch("cell_y", cell_y, "cell_y" + arrdef + "/F");
  output->Branch("cell_z", cell_z, "cell_z" + arrdef + "/F");
  output->Branch("cell_r", cell_r, "cell_r" + arrdef + "/F");
  output->Branch("cell_eta", cell_eta, "cell_eta" + arrdef + "/F");
  output->Branch("cell_theta", cell_theta, "cell_theta" + arrdef + "/F");
  output->Branch("cell_phi", cell_phi, "cell_phi" + arrdef + "/F");
  output->Branch("cell_dist", cell_dist, "cell_dist" + arrdef + "/F");
  output->Branch("cell_energy", cell_energy, "cell_energy" + arrdef + "/F");
  output->Branch("cell_wafer", cell_wafer, "cell_wafer" + arrdef + "/I");
  output->Branch("cell_wafertype", cell_wafertype, "cell_wafertype" + arrdef + "/I");
  output->Branch("cell_primary", cell_primary, "cell_primary" + arrdef + "/I");

  _input->SetBranchStatus("*", false);

  typedef std::vector<float> VFloat;
  typedef std::vector<int> VInt;
  typedef std::vector<uint32_t> VID;

  InputPtr<VFloat> cl3d_pt;
  InputPtr<VFloat> cl3d_energy;
  InputPtr<VFloat> cl3d_eta;
  InputPtr<VFloat> cl3d_phi;
  InputPtr<std::vector<VID>> cl3d_clusters_id;
  InputPtr<VInt> cl3d_showerlength;
  InputPtr<VInt> cl3d_coreshowerlength;
  InputPtr<VInt> cl3d_firstlayer;
  InputPtr<VInt> cl3d_maxlayer;
  InputPtr<VFloat> cl3d_seetot;
  InputPtr<VFloat> cl3d_seemax;
  InputPtr<VFloat> cl3d_spptot;
  InputPtr<VFloat> cl3d_sppmax;
  InputPtr<VFloat> cl3d_szz;
  InputPtr<VFloat> cl3d_srrtot;
  InputPtr<VFloat> cl3d_srrmax;
  InputPtr<VFloat> cl3d_srrmean;
  InputPtr<VFloat> cl3d_emaxe;
  InputPtr<VInt> cl3d_quality;
  InputPtr<VInt> cl3d_simtrack_pid;
  InputPtr<VFloat> cl3d_simtrack_pt;
  InputPtr<VFloat> cl3d_simtrack_eta;
  InputPtr<VFloat> cl3d_simtrack_phi;
  InputPtr<VID> tc_id;
  InputPtr<VInt> tc_layer;
  InputPtr<VInt> tc_wafer;
  InputPtr<VInt> tc_wafertype;
  InputPtr<VFloat> tc_energy;
  InputPtr<VFloat> tc_x;
  InputPtr<VFloat> tc_y;
  InputPtr<VFloat> tc_z;
  InputPtr<VInt> tc_genparticle_index;

  _input->SetBranchAddress("cl3d_pt", cl3d_pt.addr());
  _input->SetBranchAddress("cl3d_energy", cl3d_energy.addr());
  _input->SetBranchAddress("cl3d_eta", cl3d_eta.addr());
  _input->SetBranchAddress("cl3d_phi", cl3d_phi.addr());
  _input->SetBranchAddress("cl3d_clusters_id", cl3d_clusters_id.addr());
  _input->SetBranchAddress("cl3d_showerlength", cl3d_showerlength.addr());
  _input->SetBranchAddress("cl3d_coreshowerlength", cl3d_coreshowerlength.addr());
  _input->SetBranchAddress("cl3d_firstlayer", cl3d_firstlayer.addr());
  _input->SetBranchAddress("cl3d_maxlayer", cl3d_maxlayer.addr());
  _input->SetBranchAddress("cl3d_seetot", cl3d_seetot.addr());
  _input->SetBranchAddress("cl3d_seemax", cl3d_seemax.addr());
  _input->SetBranchAddress("cl3d_spptot", cl3d_spptot.addr());
  _input->SetBranchAddress("cl3d_sppmax", cl3d_sppmax.addr());
  _input->SetBranchAddress("cl3d_szz", cl3d_szz.addr());
  _input->SetBranchAddress("cl3d_srrtot", cl3d_srrtot.addr());
  _input->SetBranchAddress("cl3d_srrmax", cl3d_srrmax.addr());
  _input->SetBranchAddress("cl3d_srrmean", cl3d_srrmean.addr());
  _input->SetBranchAddress("cl3d_emaxe", cl3d_emaxe.addr());
  _input->SetBranchAddress("cl3d_quality", cl3d_quality.addr());
  _input->SetBranchAddress("cl3d_simtrack_pid", cl3d_simtrack_pid.addr());
  _input->SetBranchAddress("cl3d_simtrack_pt", cl3d_simtrack_pt.addr());
  _input->SetBranchAddress("cl3d_simtrack_eta", cl3d_simtrack_eta.addr());
  _input->SetBranchAddress("cl3d_simtrack_phi", cl3d_simtrack_phi.addr());
  _input->SetBranchAddress("tc_id", tc_id.addr());
  _input->SetBranchAddress("tc_layer", tc_layer.addr());
  _input->SetBranchAddress("tc_wafer", tc_wafer.addr());
  _input->SetBranchAddress("tc_wafertype", tc_wafertype.addr());
  _input->SetBranchAddress("tc_energy", tc_energy.addr());
  _input->SetBranchAddress("tc_x", tc_x.addr());
  _input->SetBranchAddress("tc_y", tc_y.addr());
  _input->SetBranchAddress("tc_z", tc_z.addr());
  if (_input->GetBranch("tc_genparticle_index") != nullptr)
    _input->SetBranchAddress("tc_genparticle_index", tc_genparticle_index.addr());

  constexpr double zLayer1(319.81497);
  constexpr double thetaPhiRangeMax(0.2); // arbitrary choosing 0.2x0.2 as the theta-phi bounding box

  double thetaPhiBoundaries[nBinsThetaPhi + 1];
  for (unsigned iB(0); iB <= nBinsThetaPhi; ++iB)
    thetaPhiBoundaries[iB] = -thetaPhiRangeMax + 2. * thetaPhiRangeMax / nBinsThetaPhi * iB;

  auto findThetaPhiBin([&thetaPhiBoundaries, nBinsThetaPhi](double x)->int {
      auto ub(std::upper_bound(thetaPhiBoundaries, thetaPhiBoundaries + (nBinsThetaPhi + 1), x));
      if (ub == thetaPhiBoundaries || ub == thetaPhiBoundaries + (nBinsThetaPhi + 1)) // discard out-of-bounds
        return -1;
      else
        return ub - thetaPhiBoundaries - 1;
    });

  double thetaPhiCenters[nBinsThetaPhi];
  for (unsigned iB(0); iB != nBinsThetaPhi; ++iB)
    thetaPhiCenters[iB] = (thetaPhiBoundaries[iB] + thetaPhiBoundaries[iB + 1]) * 0.5;

  long iEntry(0);
  while (iEntry != _nEntries && _input->GetEntry(iEntry++) > 0) {
    if (iEntry % 10000 == 1)
      std::cout << iEntry << std::endl;

    std::unordered_map<uint32_t, unsigned> cellMap;
    for (unsigned iT(0); iT != tc_id->size(); ++iT)
      cellMap.emplace((*tc_id)[iT], iT);

    for (unsigned iC(0); iC != cl3d_pt->size(); ++iC) {
      unsigned absPid(std::abs((*cl3d_simtrack_pid)[iC]));
      if (absPid == 0 || absPid > 1000000000)
        continue;

      if ((*cl3d_pt)[iC] < _minPt)
        continue;

      cluster_pt = (*cl3d_pt)[iC];
      cluster_eta = std::abs((*cl3d_eta)[iC]);
      cluster_phi = (*cl3d_phi)[iC];
      cluster_energy = (*cl3d_energy)[iC];
      double cluster_rho(zLayer1 / std::sinh(std::abs(cluster_eta)));
      cluster_x = cluster_rho * std::cos(cluster_phi);
      cluster_y = cluster_rho * std::sin(cluster_phi);
      cluster_zside = (cluster_eta > 0. ? 1 : -1);
      cluster_showerlength = (*cl3d_showerlength)[iC];
      cluster_coreshowerlength = (*cl3d_coreshowerlength)[iC];
      cluster_firstlayer = (*cl3d_firstlayer)[iC];
      cluster_lastlayer = (*cl3d_maxlayer)[iC];
      cluster_seetot = (*cl3d_seetot)[iC];
      cluster_seemax = (*cl3d_seemax)[iC];
      cluster_spptot = (*cl3d_spptot)[iC];
      cluster_sppmax = (*cl3d_sppmax)[iC];
      cluster_szz = (*cl3d_szz)[iC];
      cluster_srrtot = (*cl3d_srrtot)[iC];
      cluster_srrmax = (*cl3d_srrmax)[iC];
      cluster_srrmean = (*cl3d_srrmean)[iC];
      cluster_emaxe = (*cl3d_emaxe)[iC];
      cluster_quality = (*cl3d_quality)[iC];
      cluster_truth_pt = (*cl3d_simtrack_pt)[iC];
      cluster_truth_eta = std::abs((*cl3d_simtrack_eta)[iC]);
      cluster_truth_phi = (*cl3d_simtrack_phi)[iC];
      cluster_truth_energy = cluster_truth_pt * std::cosh(cluster_truth_eta);

      std::fill_n(truth_labels, nCategories, 0);

      switch (absPid) {
      case 11:
        truth_labels[kElectron] = 1;
        break;
      case 13:
        truth_labels[kMuon] = 1;
        break;
      case 22:
        truth_labels[kPhoton] = 1;
        break;
      case 111:
        truth_labels[kPi0] = 1;
        break;
      default:
        {
          unsigned q1((absPid / 1000) % 10);
          unsigned q2((absPid / 100) % 10);
          unsigned q3((absPid / 10) % 10);
          if (q1 == 0) { // meson
            if ((q2 + q3) % 2 == 0)
              truth_labels[kNeutralHadron] = 1;
            else
              truth_labels[kChargedHadron] = 1;
          }
          else {
            if ((q1 + q2 + q3) % 2 == 0) {
              if (q1 % 2 == 0 && q2 % 2 == 0) // doubly charged
                truth_labels[kChargedHadron] = 1;
              else
                truth_labels[kNeutralHadron] = 1;
            }
            else
              truth_labels[kChargedHadron] = 1;
          }
        }
        break;
      }

      for (unsigned iHit(0); iHit != maxHitsPerBin; ++iHit) {
        std::fill_n(bin_id[iHit], nBins, 0);
        std::fill_n(bin_eta[iHit], nBins, 0.);
        std::fill_n(bin_theta[iHit], nBins, 0.);
        std::fill_n(bin_phi[iHit], nBins, 0.);
        std::fill_n(bin_x[iHit], nBins, 0.);
        std::fill_n(bin_y[iHit], nBins, 0.);
        std::fill_n(bin_eta_global[iHit], nBins, 0.);
        std::fill_n(bin_theta_global[iHit], nBins, 0.);
        std::fill_n(bin_phi_global[iHit], nBins, 0.);
        std::fill_n(bin_dist_global[iHit], nBins, 0.);
        std::fill_n(bin_x_global[iHit], nBins, 0.);
        std::fill_n(bin_y_global[iHit], nBins, 0.);
        std::fill_n(bin_z_global[iHit], nBins, 0.);
        std::fill_n(bin_energy[iHit], nBins, 0.);
        std::fill_n(bin_layer[iHit], nBins, 0);
        std::fill_n(bin_wafer[iHit], nBins, 0);
        std::fill_n(bin_wafertype[iHit], nBins, 0);
      }

      auto& constituents((*cl3d_clusters_id)[iC]);

      n_cell = std::min(constituents.size(), maxHitsPerCluster);

      std::fill_n(cell_id, maxHitsPerCluster, 0);
      std::fill_n(cell_layer, maxHitsPerCluster, 0);
      std::fill_n(cell_x, maxHitsPerCluster, 0.);
      std::fill_n(cell_y, maxHitsPerCluster, 0.);
      std::fill_n(cell_z, maxHitsPerCluster, 0.);
      std::fill_n(cell_r, maxHitsPerCluster, 0.);
      std::fill_n(cell_eta, maxHitsPerCluster, 0.);
      std::fill_n(cell_theta, maxHitsPerCluster, 0.);
      std::fill_n(cell_phi, maxHitsPerCluster, 0.);
      std::fill_n(cell_dist, maxHitsPerCluster, 0.);
      std::fill_n(cell_energy, maxHitsPerCluster, 0.);
      std::fill_n(cell_wafer, maxHitsPerCluster, 0);
      std::fill_n(cell_wafertype, maxHitsPerCluster, 0);
      std::fill_n(cell_primary, maxHitsPerCluster, 0);

      double thetaMin(3.2);
      double thetaMax(0.);
      double relPhiMin(3.2);
      double relPhiMax(-3.2);

      std::map<double, int> sortedIndices;
      for (auto iS : constituents) {
        unsigned iT(cellMap.at(iS));
        sortedIndices.emplace((*tc_energy)[iT], iT);
      }

      unsigned iD(0);
      auto rEnd(sortedIndices.rend());
      for (auto rItr(sortedIndices.rbegin()); rItr != rEnd; ++rItr, ++iD) {
        if (int(iD) == n_cell) // hit maximum
          break;
        
        unsigned iT(rItr->second);

        double x((*tc_x)[iT]);
        double y((*tc_y)[iT]);
        double z(std::abs((*tc_z)[iT])); // projecting all to positive z
        double r(std::sqrt(x * x + y * y));
        double eta(std::asinh(z / r));
        double theta(std::atan2(r, z));
        double phi(std::atan2(y, x));

        cell_id[iD] = (*tc_id)[iT];
        cell_layer[iD] = (*tc_layer)[iT];
        cell_x[iD] = x;
        cell_y[iD] = y;
        cell_z[iD] = z;
        cell_r[iD] = r;
        cell_eta[iD] = eta;
        cell_theta[iD] = theta;
        cell_phi[iD] = phi;
        cell_dist[iD] = std::sqrt(r * r + z * z);
        cell_energy[iD] = (*tc_energy)[iT];
        cell_wafer[iD] = (*tc_wafer)[iT];
        cell_wafertype[iD] = (*tc_wafertype)[iT];
        if (!tc_genparticle_index->empty())
          cell_primary[iD] = ((*tc_genparticle_index)[iT] >= 0);

        double relPhi(TVector2::Phi_mpi_pi(phi - cluster_phi));

        if (theta < thetaMin)
          thetaMin = theta;
        else if (theta > thetaMax)
          thetaMax = theta;

        if (relPhi < relPhiMin)
          relPhiMin = relPhi;
        else if (relPhi > relPhiMax)
          relPhiMax = relPhi;
      }

      // center of the bounding box
      double thetaCenter((thetaMin + thetaMax) * 0.5);
      double phiCenter((relPhiMin + relPhiMax) * 0.5);

      for (unsigned iD(0); iD != constituents.size(); ++iD) {
        unsigned iT(cellMap.at(constituents[iD]));

        double x((*tc_x)[iT]);
        double y((*tc_y)[iT]);
        double z(std::abs((*tc_z)[iT]));
        double r(std::sqrt(x * x + y * y));
        double theta(std::atan2(r, z));
        double phi(std::atan2(y, x));

        int iTheta(findThetaPhiBin(theta - thetaCenter));
        if (iTheta < 0)
          continue;

        double relPhi(TVector2::Phi_mpi_pi(phi - cluster_phi));

        int iPhi(findThetaPhiBin(relPhi - phiCenter));
        if (iPhi < 0)
          continue;

        int iZ((*tc_layer)[iT]);
        if (iZ <= 27)
          iZ /= 2;
        else
          iZ -= 15;

        unsigned ihit(0);
        unsigned ibin(iZ * nBinsThetaPhi * nBinsThetaPhi + iTheta * nBinsThetaPhi + iPhi);
        // there can be at most three hits per bin
        while (ihit < maxHitsPerCluster) {
          if (bin_id[ihit][ibin] == 0)
            break;
          else
            ++ihit;
        }
        if (ihit == maxHitsPerCluster)
          continue; // discard overflow hit

        double thetaBinCenter(thetaCenter + thetaPhiCenters[iTheta]);
        double phiBinCenter(phiCenter + thetaPhiCenters[iTheta]);
        double etaBinCenter(-std::log(std::tan(0.5 * thetaBinCenter)));
        double rBinCenter(zLayer1 * std::tan(thetaBinCenter));
        double xBinCenter(rBinCenter * std::cos(phiBinCenter));
        double yBinCenter(rBinCenter * std::sin(phiBinCenter));

        double eta(std::asinh(z / r));

        bin_id[ihit][ibin] = (*tc_id)[iT];
        bin_eta[ihit][ibin] = eta - etaBinCenter;
        bin_theta[ihit][ibin] = theta - thetaBinCenter;
        bin_phi[ihit][ibin] = TVector2::Phi_mpi_pi(relPhi - phiBinCenter);
        bin_x[ihit][ibin] = x - xBinCenter;
        bin_y[ihit][ibin] = y - yBinCenter;
        bin_eta_global[ihit][ibin] = eta;
        bin_theta_global[ihit][ibin] = theta;
        bin_phi_global[ihit][ibin] = phi;
        bin_dist_global[ihit][ibin] = std::sqrt(r * r + z * z);
        bin_x_global[ihit][ibin] = x;
        bin_y_global[ihit][ibin] = y;
        bin_z_global[ihit][ibin] = z;
        bin_layer[ihit][ibin] = iZ;
        bin_energy[ihit][ibin] = (*tc_energy)[iT];
        bin_wafer[ihit][ibin] = (*tc_wafer)[iT];
        bin_wafertype[ihit][ibin] = (*tc_wafertype)[iT];
      }

      output->Fill();
    }
  }
  
  outputFile->cd();
  output->Write();
  delete outputFile;
}
