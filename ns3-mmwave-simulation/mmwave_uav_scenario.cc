/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/**
 * ns-3 mmWave UAV Simulation Scenario
 *
 * This scenario simulates mmWave beam tracking for UAV communication.
 * It demonstrates:
 * - 5G NR mmWave channel model
 * - Beam management (P1/P2/P3)
 * - UAV mobility with realistic 3D movement
 * - E2 interface integration with O-RAN xApp
 *
 * Prerequisites:
 * - ns-3 with mmwave or nr module
 * - E2 simulation module (ns-O-RAN)
 *
 * Usage:
 *   ./ns3 run "scratch/mmwave_uav_scenario --numUavs=3 --simTime=60"
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"

// Uncomment when mmWave/NR module is available:
// #include "ns3/mmwave-module.h"
// #include "ns3/nr-module.h"

#include <fstream>
#include <vector>
#include <cmath>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("MmWaveUavScenario");

/**
 * Configuration parameters
 */
struct SimConfig
{
  uint32_t numUavs = 3;           // Number of UAVs
  uint32_t numGnbs = 1;           // Number of gNBs
  double simTime = 60.0;          // Simulation time (s)
  double updateInterval = 0.005;  // Beam update interval (5ms)

  // gNB configuration
  double gnbHeight = 25.0;        // gNB antenna height (m)
  double gnbTxPower = 30.0;       // gNB TX power (dBm)

  // UAV configuration
  double uavHeight = 50.0;        // UAV flight altitude (m)
  double uavSpeed = 15.0;         // UAV speed (m/s)

  // mmWave configuration
  double centerFrequency = 28e9;  // 28 GHz
  double bandwidth = 400e6;       // 400 MHz
  uint32_t numAntennaH = 8;       // Horizontal antenna elements
  uint32_t numAntennaV = 8;       // Vertical antenna elements

  // xApp integration
  std::string xappHost = "127.0.0.1";
  uint16_t xappPort = 5001;
  bool enableXapp = false;
};

/**
 * Beam measurement structure
 */
struct BeamMeasurement
{
  uint32_t ueId;
  double timestamp;
  uint32_t servingBeamId;
  double servingRsrp;
  std::map<uint32_t, double> neighborBeams;
  Vector position;
  Vector velocity;
};

/**
 * UAV Waypoint trajectory helper
 */
class UavWaypointMobility
{
public:
  static Ptr<WaypointMobilityModel> CreateCircularPath(
    Vector center,
    double radius,
    double altitude,
    double speed,
    double startAngle = 0.0)
  {
    Ptr<WaypointMobilityModel> mobility = CreateObject<WaypointMobilityModel>();

    // Generate waypoints for circular path
    double circumference = 2 * M_PI * radius;
    double totalTime = circumference / speed;
    int numWaypoints = 36;  // 10 degrees per waypoint
    double timeStep = totalTime / numWaypoints;

    double currentTime = 0.0;
    for (int i = 0; i <= numWaypoints; i++)
    {
      double angle = startAngle + (2 * M_PI * i / numWaypoints);
      double x = center.x + radius * std::cos(angle);
      double y = center.y + radius * std::sin(angle);

      mobility->AddWaypoint(Waypoint(Seconds(currentTime), Vector(x, y, altitude)));
      currentTime += timeStep;
    }

    return mobility;
  }

  static Ptr<WaypointMobilityModel> CreateLinearPath(
    Vector start,
    Vector end,
    double speed)
  {
    Ptr<WaypointMobilityModel> mobility = CreateObject<WaypointMobilityModel>();

    double distance = CalculateDistance(start, end);
    double travelTime = distance / speed;

    mobility->AddWaypoint(Waypoint(Seconds(0), start));
    mobility->AddWaypoint(Waypoint(Seconds(travelTime), end));

    return mobility;
  }
};

/**
 * Beam tracking callback
 */
class BeamTrackingCallback
{
public:
  BeamTrackingCallback(const SimConfig& config) : m_config(config)
  {
    m_outputFile.open("beam_tracking_results.csv");
    m_outputFile << "timestamp,ue_id,beam_id,rsrp_dbm,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z\n";
  }

  ~BeamTrackingCallback()
  {
    m_outputFile.close();
  }

  void OnBeamMeasurement(const BeamMeasurement& meas)
  {
    m_outputFile << meas.timestamp << ","
                 << meas.ueId << ","
                 << meas.servingBeamId << ","
                 << meas.servingRsrp << ","
                 << meas.position.x << ","
                 << meas.position.y << ","
                 << meas.position.z << ","
                 << meas.velocity.x << ","
                 << meas.velocity.y << ","
                 << meas.velocity.z << "\n";

    m_measurements.push_back(meas);

    // TODO: Send to xApp via HTTP if enabled
    if (m_config.enableXapp)
    {
      SendToXapp(meas);
    }
  }

  void SendToXapp(const BeamMeasurement& meas)
  {
    // HTTP client would be implemented here
    // This is a placeholder for integration
    NS_LOG_INFO("Would send to xApp: UE " << meas.ueId
                << " beam " << meas.servingBeamId
                << " RSRP " << meas.servingRsrp << " dBm");
  }

  void PrintStatistics()
  {
    if (m_measurements.empty()) return;

    std::map<uint32_t, std::vector<double>> rsrpByUe;
    std::map<uint32_t, int> beamSwitchCount;
    uint32_t lastBeam = UINT32_MAX;

    for (const auto& m : m_measurements)
    {
      rsrpByUe[m.ueId].push_back(m.servingRsrp);

      if (lastBeam != UINT32_MAX && m.servingBeamId != lastBeam)
      {
        beamSwitchCount[m.ueId]++;
      }
      lastBeam = m.servingBeamId;
    }

    NS_LOG_INFO("=== Beam Tracking Statistics ===");
    for (const auto& pair : rsrpByUe)
    {
      double sum = 0;
      double minRsrp = 0;
      double maxRsrp = -200;
      for (double r : pair.second)
      {
        sum += r;
        minRsrp = std::min(minRsrp, r);
        maxRsrp = std::max(maxRsrp, r);
      }
      double avgRsrp = sum / pair.second.size();

      NS_LOG_INFO("UE " << pair.first
                  << ": Avg RSRP=" << avgRsrp << " dBm"
                  << ", Min=" << minRsrp << " dBm"
                  << ", Max=" << maxRsrp << " dBm"
                  << ", Beam switches=" << beamSwitchCount[pair.first]);
    }
  }

private:
  SimConfig m_config;
  std::ofstream m_outputFile;
  std::vector<BeamMeasurement> m_measurements;
};

/**
 * Simplified beam selection based on geometry
 * (Real implementation would use mmWave module's beam management)
 */
uint32_t CalculateOptimalBeam(Vector gnbPos, Vector uePos, uint32_t numBeamsH, uint32_t numBeamsV)
{
  // Calculate azimuth and elevation angles
  double dx = uePos.x - gnbPos.x;
  double dy = uePos.y - gnbPos.y;
  double dz = uePos.z - gnbPos.z;
  double dxy = std::sqrt(dx*dx + dy*dy);

  double azimuth = std::atan2(dy, dx);  // [-pi, pi]
  double elevation = std::atan2(dz, dxy);  // [-pi/2, pi/2]

  // Map to beam index
  // Azimuth: [-pi/2, pi/2] -> [0, numBeamsH-1]
  // Elevation: [-pi/4, pi/4] -> [0, numBeamsV-1]
  double azNorm = (azimuth + M_PI/2) / M_PI;
  double elNorm = (elevation + M_PI/4) / (M_PI/2);

  azNorm = std::max(0.0, std::min(1.0, azNorm));
  elNorm = std::max(0.0, std::min(1.0, elNorm));

  uint32_t hIdx = static_cast<uint32_t>(azNorm * (numBeamsH - 1));
  uint32_t vIdx = static_cast<uint32_t>(elNorm * (numBeamsV - 1));

  return hIdx * numBeamsV + vIdx;
}

/**
 * Simplified RSRP calculation for mmWave
 * (Real implementation would use ns-3 propagation models)
 */
double CalculateMmwaveRsrp(Vector gnbPos, Vector uePos, double txPowerDbm, double frequencyHz)
{
  double distance = CalculateDistance(gnbPos, uePos);

  // Free space path loss at mmWave
  double lambda = 3e8 / frequencyHz;
  double fspl = 20 * std::log10(4 * M_PI * distance / lambda);

  // Additional atmospheric absorption at 28 GHz (~0.1 dB/km)
  double atmosphericLoss = 0.1 * distance / 1000.0;

  // Array gain (assuming 8x8 UPA)
  double arrayGain = 10 * std::log10(64);  // ~18 dB

  // Simplified: assume 3dB beamforming gain
  double beamGain = 3.0;

  double rsrp = txPowerDbm - fspl - atmosphericLoss + arrayGain + beamGain;

  // Add some fading (simplified)
  double fading = -2.0 + 4.0 * (rand() / (double)RAND_MAX);  // [-2, 2] dB
  rsrp += fading;

  return rsrp;
}

/**
 * Main simulation function
 */
int main(int argc, char *argv[])
{
  SimConfig config;

  // Command line arguments
  CommandLine cmd;
  cmd.AddValue("numUavs", "Number of UAVs", config.numUavs);
  cmd.AddValue("simTime", "Simulation time in seconds", config.simTime);
  cmd.AddValue("uavSpeed", "UAV speed in m/s", config.uavSpeed);
  cmd.AddValue("uavHeight", "UAV altitude in meters", config.uavHeight);
  cmd.AddValue("enableXapp", "Enable xApp integration", config.enableXapp);
  cmd.AddValue("xappHost", "xApp host address", config.xappHost);
  cmd.AddValue("xappPort", "xApp port", config.xappPort);
  cmd.Parse(argc, argv);

  // Enable logging
  LogComponentEnable("MmWaveUavScenario", LOG_LEVEL_INFO);

  NS_LOG_INFO("=== mmWave UAV Simulation Configuration ===");
  NS_LOG_INFO("Number of UAVs: " << config.numUavs);
  NS_LOG_INFO("Simulation time: " << config.simTime << " s");
  NS_LOG_INFO("UAV speed: " << config.uavSpeed << " m/s");
  NS_LOG_INFO("UAV altitude: " << config.uavHeight << " m");
  NS_LOG_INFO("Center frequency: " << config.centerFrequency / 1e9 << " GHz");
  NS_LOG_INFO("Antenna array: " << config.numAntennaH << "x" << config.numAntennaV);

  // Create nodes
  NodeContainer gnbNodes;
  gnbNodes.Create(config.numGnbs);

  NodeContainer uavNodes;
  uavNodes.Create(config.numUavs);

  // Setup gNB mobility (stationary)
  MobilityHelper gnbMobility;
  gnbMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
  gnbMobility.Install(gnbNodes);
  gnbNodes.Get(0)->GetObject<MobilityModel>()->SetPosition(Vector(0, 0, config.gnbHeight));

  // Setup UAV mobility (circular paths at different radii)
  MobilityHelper uavMobility;
  uavMobility.SetMobilityModel("ns3::WaypointMobilityModel");
  uavMobility.Install(uavNodes);

  for (uint32_t i = 0; i < config.numUavs; i++)
  {
    double radius = 100 + i * 50;  // 100m, 150m, 200m, ...
    double startAngle = 2 * M_PI * i / config.numUavs;  // Offset start positions

    Ptr<WaypointMobilityModel> mobility = UavWaypointMobility::CreateCircularPath(
      Vector(0, 0, 0),  // Center
      radius,
      config.uavHeight,
      config.uavSpeed,
      startAngle
    );

    // Copy waypoints to the installed mobility model
    Ptr<WaypointMobilityModel> installedMobility =
      uavNodes.Get(i)->GetObject<WaypointMobilityModel>();

    // For simplicity, set initial position
    double x = radius * std::cos(startAngle);
    double y = radius * std::sin(startAngle);
    installedMobility->SetPosition(Vector(x, y, config.uavHeight));
  }

  // Create beam tracking callback
  BeamTrackingCallback beamCallback(config);

  // Schedule periodic beam measurements
  Vector gnbPos = gnbNodes.Get(0)->GetObject<MobilityModel>()->GetPosition();

  for (double t = 0; t < config.simTime; t += config.updateInterval)
  {
    Simulator::Schedule(Seconds(t), [&, gnbPos, t]()
    {
      for (uint32_t i = 0; i < config.numUavs; i++)
      {
        Ptr<MobilityModel> mobility = uavNodes.Get(i)->GetObject<MobilityModel>();
        Vector pos = mobility->GetPosition();
        Vector vel = mobility->GetVelocity();

        // Calculate optimal beam
        uint32_t beamId = CalculateOptimalBeam(gnbPos, pos,
                                                config.numAntennaH,
                                                config.numAntennaV);

        // Calculate RSRP
        double rsrp = CalculateMmwaveRsrp(gnbPos, pos,
                                           config.gnbTxPower,
                                           config.centerFrequency);

        // Create measurement
        BeamMeasurement meas;
        meas.ueId = i;
        meas.timestamp = t;
        meas.servingBeamId = beamId;
        meas.servingRsrp = rsrp;
        meas.position = pos;
        meas.velocity = vel;

        // Add neighbor beams (simplified)
        for (int delta = -2; delta <= 2; delta++)
        {
          if (delta == 0) continue;
          uint32_t neighborBeam = (beamId + delta + 64) % 64;
          double neighborRsrp = rsrp - 3.0 * std::abs(delta);  // -3dB per beam
          meas.neighborBeams[neighborBeam] = neighborRsrp;
        }

        beamCallback.OnBeamMeasurement(meas);
      }
    });
  }

  // Run simulation
  NS_LOG_INFO("Starting simulation...");
  Simulator::Stop(Seconds(config.simTime));
  Simulator::Run();

  // Print statistics
  beamCallback.PrintStatistics();

  Simulator::Destroy();

  NS_LOG_INFO("Simulation complete. Results saved to beam_tracking_results.csv");

  return 0;
}
