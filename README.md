# LEO Constellation Resource Optimization & RF Link Simulation

This repository contains the code and documentation for our final project in the **Satellite and Mobile Communication** course (under Prof. Shlomi Arnon) at **Ben-Gurion University of the Negev**.

## Project Overview
This project focuses on the physical RF layer simulation of satellite communication links and evaluates network-level resource allocation schemes. It is heavily inspired by and provides a practical implementation of the concepts presented in the following academic paper:

> **Academic Credit:**
> Xu Ma, Haijun Zhang, Wei Song, Yang Lu, Yuan Wu, and Victor C. M. Leung, *"Resource Optimization for LEO Constellation Networks: A Multi-Satellite Cooperative Coverage Design,"* IEEE Transactions on Communications, 2025

We simulate the RF physical layer (using QAM-16, TWTA nonlinearity, and AWGN) and overlay the network-layer improvements proposed in the paper. The goal is to evaluate how mitigating co-layer interference translates into measurable improvements in Bit Error Rate (BER) and theoretical network throughput.

## Repository Structure

* 'part1_settlite.py': **RF Satellite Link Simulation**. 
  Models RF impairments end-to-end, including:
  * HPA nonlinearity (Saleh TWTA).
  * Free-space path loss.
  * Doppler shift.
  * Receiver thermal noise (AWGN).
  * Phase noise (1/f).
  * I/Q amplitude & phase imbalance and DC offsets.

* 'part3_settlite.py': **LEO Constellation Resource Optimization**. 
  Simulates the physical layer and overlays the paper's network improvements:
  * **Co-layer interference mitigation:** Comparing a Baseline (K-means) scheme with high interference to the Proposed Cooperative Coverage Scheme (CCS) which utilizes an Improved P-center algorithm.
  * **EPFD constraint management:** Ensuring the LEO constellation actively manages power and maintains strict compliance with ITU EPFD limits (-205 dBW) to protect legacy GSO networks.

## Authors
* **Shahar Lavi**
* **Itamar Guerchon**
