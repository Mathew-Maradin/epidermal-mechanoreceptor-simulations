# Passive Transduction Features of Skin

### Project Overview

The human skin serves as a complex interface that modulates external mechanical stimuli before they reach sensory neurons. This project explores how structural features—specifically epidermal ridges (fingerprints)—passively enhance tactile transduction. By utilizing computational simulations in Python, the study evaluates the relationship between skin geometry and the activation of four primary mechanoreceptor types (SAI, SAII, FAI, and FAII) under various pressure conditions.

### Preprocessing

**Data Pipeline:**

* **Mechanical Modeling:** The skin is modeled as a **Kelvin-Voigt** viscoelastic material, combining a viscous dashpot and an elastic spring in parallel to simulate the skin's "creep" and deformation over time.
* **Geometric Profiling:** Three distinct surface structures were generated to represent different fingerprint patterns:
* **Flat:** A uniform profile used as a control.
* **Wave-like (Arches):** Generated using a sine function with added **Perlin noise** for natural variation.
* **Spiral (Whorls/Loops):** Created using varying frequencies and phase shifts to mimic complex ridge curvatures.


* **Pressure Mapping:** Applied external forces through circular, point-based, and linear pressure distributions to observe how different force geometries interact with skin ridges.

### Models & Results

**1. Slow-Adapting Receptors (SAI & SAII)**

* **Logic:** SAI (Merkel disks) detect sustained pressure, while SAII (Ruffini endings) respond to skin stretching.
* **Response:** In simulations, these receptors showed sustained activity as pressure stabilized.
* **Insight:** Ridged surfaces (spiral and wave) significantly amplified these signals compared to flat surfaces, as the local curvature increases the mechanical strain on the receptor.

**2. Fast-Adapting Receptors (FAI & FAII)**

* **Logic:** FAI (Meissner corpuscles) detect dynamic deformation (changes in stimuli), while FAII (Pacinian corpuscles) respond to high-frequency vibrations.
* **Response:** FAI responses peaked during the initial application of pressure and decreased as it stabilized. FAII responses showed a sharp, discrete spike at the moment of impact.
* **Observation:** The spiral ridge pattern provided the highest signal amplification for FA receptors due to its omnidirectional curvature.

**3. Parametric Variations (Young’s Modulus & Viscosity)**

* **Young’s Modulus (1.0–2.0 MPa):** Stiffer skin (higher modulus) resulted in lower overall deformation and reduced receptor responses, matching theoretical expectations for skin aging and hydration.
* **Viscosity Coefficient (0.1–1.0 Paᐧsec):** Lower viscosity resulted in higher receptor magnitudes, as the material provided less dampening for the mechanical signals.

### Summary

The simulation successfully validates that fingerprint ridges act as mechanical filters that amplify stress distribution. By incorporating the Kelvin-Voigt model and varying geometric profiles, the project demonstrates that ridge curvature is a vital "passive" feature that enhances the sensitivity of tactile encoding. While the FAII model showed minor numerical artifacts (negative values) at very low viscosity, the overall results align closely with biological touch theory, confirming that our ability to perceive fine textures is largely dependent on the physical structure of our skin.