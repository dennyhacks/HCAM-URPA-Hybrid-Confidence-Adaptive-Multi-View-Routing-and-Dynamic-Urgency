"""
rail_data.py - Data Simulation and EDA for Rail Madad (SIH1711)
Course: Data Science Ecosystem (23CS2024)

This script synthesizes `rail_madad_sim.csv` adhering strictly to course requirements:
- 8 interpretable photo features (rust, dirt, damage, brightness, blurriness, crowd, fire_smoke, sparks)
- ~35% missing photos (features set to NaN)
- ~25% vague/ambiguous complaint texts
- Fake PNR, passenger name, and phone columns for data ethics/anonymization
- Operational telemetry and ground-truth physics-based urgency parameters
"""

import os
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Ground truth physical parameters for Dynamic Urgency Propagation (URPA)
A_TRUE = {
    "Mechanical": 0.85,
    "Electrical": 0.65,
    "Catering": 0.35,
    "Cleaning": 0.45,
    "Security": 0.90,
}
TAU_TRUE_KM = 60.0  # Yard proximity decay constant (km)
B_TRUE = 0.08       # Passenger wait time hazard rate (1/hour)

DEPARTMENTS = ["Mechanical", "Electrical", "Catering", "Cleaning", "Security"]
DEPT_WEIGHTS = [0.32, 0.24, 0.14, 0.20, 0.10]

STATIONS = [
    "NDLS", "BCT", "HWH", "MAS", "SBC", "CNB", "BSB",
    "PNBE", "PUNE", "ADI", "GKP", "HYB", "CSMT", "TVC", "AGC"
]

TEMPLATES = {
    "Mechanical": [
        "Coach {coach} berth {berth} emergency window latch broken and rattling at high speed",
        "Coupling jerk felt severely in coach {coach}, metallic grinding noise from bogie wheelset",
        "Berth {berth} upper frame cracked and tilting downwards in coach {coach}",
        "Vestibule door between {coach} and next coach jammed shut cannot open",
        "Air conditioner compressor failed in coach {coach}, suffocating atmosphere and no cooling",
        "Window pane shattered by stone pelting near outer signal, glass shards on seat {berth}",
        "Coach {coach} suspension making loud thumping sound on curves, severe vibrations",
        "Upper berth chain snapped at seat {berth} coach {coach}, serious safety hazard",
        "Bathroom door latch broken inside coach {coach}, door swinging open continuously",
        "Water tank leaking heavily under coach {coach} bogie frame near axle box",
    ],
    "Electrical": [
        "Berth {berth} mobile charging socket completely dead and sparking in coach {coach}",
        "Coach {coach} all ceiling fans stopped working since past two hours in peak heat",
        "Main switchboard in coach {coach} smelling of burning plastic and emitting smoke",
        "Reading lamp wires exposed with sparks near passenger berth {berth} coach {coach}",
        "No lights inside entire coach {coach}, running in pitch dark at night",
        "Air conditioning vents blowing hot air instead of cool air in coach {coach}",
        "Voltage fluctuation causing fans to run at crawling speed in coach {coach}",
        "Exhaust fan in toilet not operating causing terrible air circulation in coach {coach}",
        "Emergency lights not functioning during night halt in coach {coach}",
        "Charging point at berth {berth} shorted out and tripped circuit breaker",
    ],
    "Catering": [
        "Pantry staff demanding Rs 150 for standard Rs 80 vegetarian thali in coach {coach}",
        "Dinner served in coach {coach} berth {berth} has spoiled sour dal and bad smell",
        "Pantry vendor refused to provide official printed bill for tea and water bottle",
        "Unhygienic food handling in pantry car, cockroaches seen crawling near food containers",
        "Packaged drinking water bottle seal was broken and bottle filled with tap water",
        "Breakfast omitted and not served to passengers with confirmed pre-paid meal booking",
        "Stale bread and fungus on cutlets provided by IRCTC pantry vendor in coach {coach}",
        "Overcharging on snacks and water bottles above maximum retail price in train",
        "Pantry staff misbehaving rudely when asked for menu card and pricing list",
        "Vegetarian meal contaminated with egg pieces served to passenger at berth {berth}",
    ],
    "Cleaning": [
        "Coach {coach} western toilet choked with waste overflowing into aisle doorway",
        "Severe foul stench from washroom making breathing impossible near berth {berth}",
        "Floor of coach {coach} littered with food packets and plastic bottles, no sweeper since morning",
        "No water in washbasin and toilet flush tanks throughout coach {coach}",
        "Cockroaches and bedbugs infesting berth {berth} in coach {coach}, unable to sleep",
        "Washbasin drain completely blocked with dirty water splashing on luggage",
        "Liquid soap dispenser empty and dirty washbasin mirrors in coach {coach}",
        "Dustbins overflowing in coach {coach} doorway attracting flies and rodents",
        "Used bedrolls from previous trip dumped near doorway in coach {coach}",
        "Toilet floor covered in muddy slurry, extremely slippery and hazardous for elderly",
    ],
    "Security": [
        "Unauthorized passengers entered coach {coach} without ticket and occupying reserved berths",
        "Passenger luggage stolen from under berth {berth} while train stopped at outer signal",
        "Drunk individuals harassing female passengers and using abusive language in coach {coach}",
        "Emergency alarm chain pulled repeatedly by miscreants between stations",
        "Suspicious unattended black backpack left under seat {berth} for past two hours",
        "Physical scuffle and fighting among youth groups inside compartment {coach}",
        "Beggars and unauthorized hawkers continuously disturbing passengers inside coach {coach}",
        "Smell of contraband smoking inside toilet of coach {coach}, passengers feel threatened",
        "Stowaways sitting on vestibule floor blocking bathroom and threatening passengers",
        "Pickpocket caught red-handed near doorway in coach {coach}, crowd becoming agitated",
    ],
}

# ~25% ambiguous complaints without explicit departmental vocabulary
AMBIGUOUS_TEMPLATES = [
    "Problem in coach {coach}, please send railway staff immediately",
    "Issue reported near berth {berth} in coach {coach}, kindly attend as soon as possible",
    "Terrible experience in coach {coach}, urgent action needed from authorities",
    "Condition in coach {coach} berth {berth} is completely unacceptable",
    "Assistance required in coach {coach} as soon as possible, please help",
    "Passengers facing severe trouble in coach {coach}, kindly send inspection team",
    "Please inspect coach {coach} berth {berth} at the upcoming station halt",
    "Formal complaint regarding coach {coach}, immediate resolution requested",
    "Unsatisfactory situation in coach {coach} near berth {berth}",
    "Requesting railway staff to check coach {coach} berth {berth} immediately",
]

COACH_PREFIXES = ["S1", "S2", "S3", "S4", "S5", "B1", "B2", "B3", "B4", "A1", "A2", "H1", "GEN1", "GEN2"]

FIRST_NAMES = [
    "Ramesh", "Suresh", "Pooja", "Priya", "Amit", "Rahul", "Sneha", "Ananya", "Vikram",
    "Deepak", "Sunita", "Rajesh", "Kavita", "Manoj", "Arun", "Divya", "Rohit", "Neha",
    "Karthik", "Sanjay", "Anil", "Meena", "Gaurav", "Swati", "Naveen", "Jyoti", "Harish"
]
LAST_NAMES = [
    "Sharma", "Verma", "Patel", "Singh", "Kumar", "Nair", "Iyer", "Reddy", "Rao",
    "Gupta", "Mishra", "Joshi", "Deshmukh", "Choudhury", "Bose", "Banerjee", "Das",
    "Menon", "Pillai", "Yadav", "Pandey", "Agarwal", "Kulkarni", "Ghosh", "Tripathi"
]

# Photo presence probability per department (~35% missing overall)
IMAGE_PROBS = {
    "Mechanical": 0.68,
    "Electrical": 0.62,
    "Catering": 0.58,
    "Cleaning": 0.78,
    "Security": 0.55,
}

# Base severity priors per department (mean, std)
SEVERITY_PRIORS = {
    "Mechanical": (0.65, 0.12),
    "Electrical": (0.50, 0.14),
    "Catering": (0.35, 0.12),
    "Cleaning": (0.42, 0.13),
    "Security": (0.78, 0.12),
}

# Department-specific means for the 8 interpretable photo features:
# [rust_score, dirt_score, damage_score, brightness, blurriness, crowd_density, fire_smoke_risk, sparks_wiring_risk]
PHOTO_FEATURE_PROFILES = {
    "Mechanical": [0.72, 0.35, 0.78, 0.55, 0.28, 0.20, 0.10, 0.15],
    "Electrical": [0.20, 0.25, 0.40, 0.60, 0.30, 0.22, 0.48, 0.82],
    "Catering":   [0.10, 0.65, 0.25, 0.62, 0.25, 0.55, 0.08, 0.05],
    "Cleaning":   [0.30, 0.88, 0.35, 0.45, 0.40, 0.25, 0.05, 0.05],
    "Security":   [0.15, 0.30, 0.38, 0.48, 0.52, 0.82, 0.20, 0.12],
}


def generate_dataset(num_samples: int = 3500) -> pd.DataFrame:
    """Synthesizes multi-modal Rail Madad complaints with interpretable photo features and ethics fields."""
    records = []
    base_time = datetime(2026, 3, 1, 6, 0, 0)

    for i in range(num_samples):
        dept = np.random.choice(DEPARTMENTS, p=DEPT_WEIGHTS)
        coach = random.choice(COACH_PREFIXES)
        berth = random.randint(1, 72)

        # ~25% ambiguous text distribution
        is_ambiguous = random.random() < 0.25
        if is_ambiguous:
            template = random.choice(AMBIGUOUS_TEMPLATES)
        else:
            template = random.choice(TEMPLATES[dept])
        complaint_text = template.format(coach=coach, berth=berth)

        station_code = random.choice(STATIONS)
        random_minutes = random.randint(0, 45 * 24 * 60)
        timestamp = (base_time + timedelta(minutes=random_minutes)).strftime("%Y-%m-%d %H:%M:%S")

        train_speed_kmph = float(np.round(np.random.uniform(0.0, 125.0), 1))
        dist_to_yard_km = float(np.round(np.random.uniform(3.0, 160.0), 1))
        wait_hours = float(np.round(np.random.exponential(scale=2.5) + 0.1, 2))
        wait_hours = min(wait_hours, 18.0)

        # Photo presence & 8 interpretable features
        has_image = 1 if (random.random() < IMAGE_PROBS[dept]) else 0
        if has_image == 1:
            profile = PHOTO_FEATURE_PROFILES[dept]
            # Sample with Gaussian perturbation bounded in [0.0, 1.0]
            rust_score = float(np.clip(np.random.normal(profile[0], 0.12), 0.0, 1.0))
            dirt_score = float(np.clip(np.random.normal(profile[1], 0.12), 0.0, 1.0))
            damage_score = float(np.clip(np.random.normal(profile[2], 0.12), 0.0, 1.0))
            brightness = float(np.clip(np.random.normal(profile[3], 0.10), 0.0, 1.0))
            blurriness = float(np.clip(np.random.normal(profile[4], 0.12), 0.0, 1.0))
            crowd_density = float(np.clip(np.random.normal(profile[5], 0.12), 0.0, 1.0))
            fire_smoke_risk = float(np.clip(np.random.normal(profile[6], 0.10), 0.0, 1.0))
            sparks_wiring_risk = float(np.clip(np.random.normal(profile[7], 0.12), 0.0, 1.0))
        else:
            # Missing when no photo
            rust_score = np.nan
            dirt_score = np.nan
            damage_score = np.nan
            brightness = np.nan
            blurriness = np.nan
            crowd_density = np.nan
            fire_smoke_risk = np.nan
            sparks_wiring_risk = np.nan

        # Base severity score
        sev_mean, sev_std = SEVERITY_PRIORS[dept]
        sev = float(np.clip(np.random.normal(sev_mean, sev_std), 0.05, 0.98))

        # Dynamic Urgency score using true physical dynamics
        a_k = A_TRUE[dept]
        hazard = a_k * np.exp(-dist_to_yard_km / TAU_TRUE_KM) + B_TRUE * wait_hours
        urg = sev + (1.0 - sev) * (1.0 - np.exp(-hazard))
        urg = float(np.clip(urg + np.random.normal(0, 0.02), 0.05, 0.99))

        # Fake PII for Ethics Module (Anonymisation)
        passenger_name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        passenger_phone = f"+91 {random.randint(70000, 99999)} {random.randint(10000, 99999)}"
        pnr_number = f"{random.randint(100, 999)}-{random.randint(1000000, 9999999)}"

        records.append({
            "complaint_id": f"RM-2026-{10000 + i}",
            "passenger_name": passenger_name,
            "passenger_phone": passenger_phone,
            "pnr_number": pnr_number,
            "station_code": station_code,
            "timestamp": timestamp,
            "complaint_text": complaint_text,
            "train_speed_kmph": train_speed_kmph,
            "dist_to_yard_km": dist_to_yard_km,
            "wait_hours": wait_hours,
            "image_present": has_image,
            "rust_score": round(rust_score, 4) if not np.isnan(rust_score) else np.nan,
            "dirt_score": round(dirt_score, 4) if not np.isnan(dirt_score) else np.nan,
            "damage_score": round(damage_score, 4) if not np.isnan(damage_score) else np.nan,
            "brightness": round(brightness, 4) if not np.isnan(brightness) else np.nan,
            "blurriness": round(blurriness, 4) if not np.isnan(blurriness) else np.nan,
            "crowd_density": round(crowd_density, 4) if not np.isnan(crowd_density) else np.nan,
            "fire_smoke_risk": round(fire_smoke_risk, 4) if not np.isnan(fire_smoke_risk) else np.nan,
            "sparks_wiring_risk": round(sparks_wiring_risk, 4) if not np.isnan(sparks_wiring_risk) else np.nan,
            "true_department": dept,
            "severity_score": round(sev, 4),
            "urgency_score": round(urg, 4),
            "is_ambiguous": int(is_ambiguous),
        })

    df = pd.DataFrame(records)
    return df


def run_eda(df: pd.DataFrame, output_dir: str = ".") -> None:
    """Computes descriptive EDA statistics and saves figures."""
    print("=" * 70)
    print("EXPLORATORY DATA ANALYSIS (EDA) - RAIL MADAD SIMULATED DATASET")
    print("=" * 70)
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
    print("Missing Values Analysis:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    ambig_count = df["is_ambiguous"].sum()
    print(f"\nAmbiguous Complaint Texts: {ambig_count} ({ambig_count / len(df) * 100:.1f}%)")

    print("\nDepartment Distribution (Class Imbalance):")
    counts = df["true_department"].value_counts()
    for dept, count in counts.items():
        pct = (count / len(df)) * 100
        print(f"  - {dept:<12}: {count:>5} ({pct:.1f}%)")

    total_missing_imgs = (df["image_present"] == 0).sum()
    print(f"\nComplaints Without Photos: {total_missing_imgs} ({total_missing_imgs / len(df) * 100:.1f}%)")

    # Save figures
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    palette = sns.color_palette("Set2", len(DEPARTMENTS))

    sns.countplot(data=df, x="true_department", hue="true_department", legend=False, order=DEPARTMENTS, ax=axes[0], palette=palette)
    axes[0].set_title("Complaint Count by Department (Class Imbalance)", fontsize=12, fontweight="bold")
    axes[0].grid(axis="y", linestyle="--", alpha=0.6)

    img_df = df.groupby("true_department")["image_present"].mean().reset_index()
    sns.barplot(data=img_df, x="true_department", y="image_present", hue="true_department", legend=False, order=DEPARTMENTS, ax=axes[1], palette=palette)
    axes[1].set_title("Photo Attachment Rate by Department", fontsize=12, fontweight="bold")
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig1_path = os.path.join(output_dir, "eda_department_dist.png")
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    print(f"[Saved EDA Plot 1] -> {fig1_path}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    scatter1 = axes[0].scatter(df["dist_to_yard_km"], df["urgency_score"], c=df["severity_score"], cmap="viridis", alpha=0.6, s=18)
    plt.colorbar(scatter1, ax=axes[0], label="Base Severity Score")
    axes[0].set_title("Urgency vs Distance to Yard (km)", fontsize=12, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    scatter2 = axes[1].scatter(df["wait_hours"], df["urgency_score"], c=df["severity_score"], cmap="plasma", alpha=0.6, s=18)
    plt.colorbar(scatter2, ax=axes[1], label="Base Severity Score")
    axes[1].set_title("Urgency vs Waiting Time (Hours)", fontsize=12, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig2_path = os.path.join(output_dir, "eda_urgency_vs_dist_wait.png")
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    print(f"[Saved EDA Plot 2] -> {fig2_path}")


def main():
    csv_file = "rail_madad_sim.csv"
    print("Generating Rail Madad dataset with 8 photo features and ethics columns...")
    df = generate_dataset(num_samples=3500)
    df.to_csv(csv_file, index=False)
    print(f"Dataset saved to '{csv_file}' ({len(df)} rows).")
    run_eda(df)


if __name__ == "__main__":
    main()
