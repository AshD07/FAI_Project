# Stage 10: Performance Evaluation Report

Scenarios: 5 | Agents: 4 | Runs per agent per scenario: 5 | Total trials: 100 | Lives per ship: 1

## 1. Headline Results (computed from the data below)
| Measure | Best agent | Value |
| :--- | :--- | :--- |
| Mean score | Naive Reactive | 182.4 |
| Mean shooting accuracy | Autonomous Pilot (GA-Optimized) | 40.1% |
| Mean survival time | Autonomous Pilot (Default) | 20.2 s |

## 2. Scenario Benchmark Results

### Scenario: `1_Target_Accuracy`
| Agent | Score | Hits | Accuracy | Deaths | Survival Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Autonomous Pilot (Default)** | 168.0 ± 71.6 | 5.4 ± 2.9 | 43.1% ± 32.9% | 0.6 ± 0.5 / 1 | 19.5 ± 9.7s |
| **Autonomous Pilot (GA-Optimized)** | 187.0 ± 86.7 | 7.2 ± 4.2 | 44.0% ± 32.0% | 0.8 ± 0.4 / 1 | 22.9 ± 8.6s |
| **Naive Reactive** | 266.0 ± 154.2 | 13.2 ± 8.9 | 16.2% ± 6.2% | 0.8 ± 0.4 / 1 | 14.6 ± 11.6s |
| **Random Baseline** | 9.0 ± 20.1 | 0.6 ± 1.3 | 4.0% ± 8.9% | 0.4 ± 0.5 / 1 | 23.8 ± 12.0s |

### Scenario: `2_Hazard_Swarm`
| Agent | Score | Hits | Accuracy | Deaths | Survival Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Autonomous Pilot (Default)** | 164.0 ± 94.4 | 6.4 ± 4.6 | 40.8% ± 16.5% | 0.6 ± 0.5 / 1 | 21.7 ± 7.7s |
| **Autonomous Pilot (GA-Optimized)** | 138.0 ± 84.3 | 6.2 ± 4.4 | 53.0% ± 29.1% | 0.4 ± 0.5 / 1 | 21.8 ± 11.7s |
| **Naive Reactive** | 217.0 ± 67.6 | 10.2 ± 4.0 | 25.0% ± 6.2% | 1.0 ± 0.0 / 1 | 7.0 ± 2.8s |
| **Random Baseline** | 29.0 ± 47.7 | 1.2 ± 1.8 | 5.6% ± 11.8% | 0.6 ± 0.5 / 1 | 14.4 ± 14.3s |

### Scenario: `3_Passive_Rival`
| Agent | Score | Hits | Accuracy | Deaths | Survival Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Autonomous Pilot (Default)** | 254.0 ± 155.7 | 6.4 ± 3.4 | 29.4% ± 7.2% | 0.2 ± 0.4 / 1 | 26.4 ± 7.9s |
| **Autonomous Pilot (GA-Optimized)** | 129.0 ± 51.5 | 4.8 ± 2.5 | 47.2% ± 35.3% | 0.6 ± 0.5 / 1 | 18.3 ± 11.9s |
| **Naive Reactive** | 153.0 ± 136.8 | 4.8 ± 2.6 | 10.6% ± 3.7% | 1.0 ± 0.0 / 1 | 7.6 ± 3.0s |
| **Random Baseline** | 9.0 ± 15.2 | 0.6 ± 0.9 | 2.8% ± 5.9% | 0.4 ± 0.5 / 1 | 23.8 ± 12.0s |

### Scenario: `4_Adversarial_Duel`
| Agent | Score | Hits | Accuracy | Deaths | Survival Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Autonomous Pilot (Default)** | 227.0 ± 192.4 | 6.4 ± 4.5 | 29.7% ± 19.8% | 0.8 ± 0.4 / 1 | 18.8 ± 11.8s |
| **Autonomous Pilot (GA-Optimized)** | 124.0 ± 101.8 | 5.4 ± 5.0 | 37.8% ± 11.5% | 0.8 ± 0.4 / 1 | 13.5 ± 10.3s |
| **Naive Reactive** | 95.0 ± 46.5 | 4.2 ± 2.7 | 13.5% ± 5.9% | 1.0 ± 0.0 / 1 | 4.8 ± 1.3s |
| **Random Baseline** | 16.0 ± 22.2 | 1.0 ± 1.4 | 3.5% ± 5.8% | 0.8 ± 0.4 / 1 | 15.2 ± 12.6s |

### Scenario: `5_High_Speed_Debris`
| Agent | Score | Hits | Accuracy | Deaths | Survival Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Autonomous Pilot (Default)** | 50.0 ± 63.9 | 2.4 ± 2.9 | 19.9% ± 19.3% | 0.8 ± 0.4 / 1 | 14.7 ± 11.0s |
| **Autonomous Pilot (GA-Optimized)** | 34.0 ± 36.0 | 1.4 ± 1.3 | 18.3% ± 17.1% | 0.8 ± 0.4 / 1 | 14.3 ± 11.0s |
| **Naive Reactive** | 181.0 ± 64.0 | 8.4 ± 4.0 | 14.9% ± 3.5% | 1.0 ± 0.0 / 1 | 9.0 ± 2.4s |
| **Random Baseline** | 9.0 ± 15.2 | 0.6 ± 0.9 | 0.7% ± 1.0% | 1.0 ± 0.0 / 1 | 12.5 ± 10.3s |

## 3. Overall Agent Performance Averages
| Agent Name | Mean Score | Mean Accuracy | Mean Survival (s) | Death Rate | Emergency-Evasion Ticks |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Baseline** | 14.4 ± 25.9 | 3.3% | 17.9 ± 12.3 | 64% | 0.0 |
| **Naive Reactive** | 182.4 ± 111.2 | 16.1% | 8.6 ± 6.1 | 96% | 0.0 |
| **Autonomous Pilot (Default)** | 172.6 ± 135.6 | 32.6% | 20.2 ± 9.8 | 60% | 379.1 |
| **Autonomous Pilot (GA-Optimized)** | 122.4 ± 85.9 | 40.1% | 18.2 ± 10.6 | 68% | 336.7 |

Notes: accuracy = hits / shots fired; survival time is capped at the 30 s scenario limit and a match also ends early if every asteroid is destroyed; emergency-evasion ticks count frames on which the reflexive dodge rule controlled the ship (not distinct dodges).