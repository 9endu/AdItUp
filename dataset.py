"""
dataset.py — Synthetic UAE classified ads dataset generator
"""

import pandas as pd
import random

SAMPLES = {
    "cars": [
        ("Toyota Camry 2020 for sale", "Well maintained Toyota Camry 2020, low mileage 45000 km, GCC spec, single owner, full service history, accident free, price negotiable"),
        ("Nissan Patrol 2019 V8", "Nissan Patrol 2019 V8 Platinum edition, GCC specs, sunroof, leather seats, 7 seater, 80000 km, excellent condition"),
        ("Honda Civic 2021", "Honda Civic 2021 model, 1.5 turbo, low mileage 30000 km, original paint, no accidents, full service Al-Futtaim"),
        ("BMW 3 Series 2018 for sale", "BMW 320i 2018, expat owned, service history with agency, sunroof, parking sensors, reverse camera, 55000 km"),
        ("Mitsubishi Pajero 2017 4x4", "Pajero 2017 full option, 4x4, V6, 7 seats, good condition, new tyres, passed RTA, urgent sale due to relocation"),
        ("Hyundai Tucson 2022 almost new", "Hyundai Tucson 2022, only 15000 km, under warranty, apple carplay, lane assist, backup camera, GCC specs"),
        ("Kia Sportage 2020 top model", "Kia Sportage 2020, panoramic roof, leather seats, all wheel drive, full service at agency, excellent condition, AED 55000"),
        ("Ford F150 pickup truck 2019", "Ford F150 2019, Lariat trim, 4x4, V8 engine, towing package, tonneau cover, very clean, single owner"),
        ("Chevrolet Malibu 2018 automatic", "Chevrolet Malibu LT 2018, 2.0 turbo, automatic, leather interior, remote start, 70000 km, accident free"),
        ("Lexus RX350 2016 full option", "Lexus RX350 2016, full option, navigation, 360 camera, heated seats, GCC, one owner, service history at Lexus UAE"),
        ("Mercedes C200 2017 for sale", "Mercedes Benz C200 2017, panoramic sunroof, paddle shifters, full leather, low mileage, agency maintained, AED 75000"),
        ("Range Rover Evoque 2018", "Range Rover Evoque 2018, HSE trim, AWD, touch screen, meridian sound system, 60000 km, service history available"),
        ("Mazda CX5 2021 turbo", "Mazda CX5 2021 turbo, signature edition, all wheel drive, heads up display, 20000 km, under warranty, AED 95000"),
        ("Jeep Wrangler 2020 Rubicon", "Jeep Wrangler Rubicon 2020, hard and soft top, lifted, off road tyres, 35000 km, great condition, weekend warrior"),
        ("Toyota Land Cruiser 2015", "Land Cruiser GXR 2015, V8 diesel, 7 seater, electric sunroof, 120000 km, well maintained, AED 125000 negotiable"),
    ],

    "property": [
        ("2 bedroom apartment for rent in Dubai Marina", "Spacious 2 bedroom apartment available in Dubai Marina, sea view, fully furnished, gym and pool access, parking included, AED 95000 per year"),
        ("Studio flat for rent in Deira Dubai", "Clean studio apartment in Deira, close to metro, shared kitchen, bills included, suitable for working professionals, AED 28000 yearly"),
        ("3 bedroom villa for sale in Abu Dhabi", "3 bedroom villa in Al Raha Gardens Abu Dhabi, private garden, maid room, 2 parking spaces, community pool, AED 2.2 million"),
        ("Office space for rent in Business Bay", "800 sqft fitted office in Business Bay, floor to ceiling windows, reception area, meeting room, 2 parking spaces, AED 80000 per year"),
        ("1 bedroom apartment Jumeirah Village Circle", "Brand new 1 bedroom in JVC, kitchen appliances included, balcony, building amenities, close to supermarket, flexible payments"),
        ("Penthouse for sale Palm Jumeirah", "Luxury penthouse on Palm Jumeirah, 4 bedrooms, private pool, full sea view, 4 parking, smart home system, AED 12 million"),
        ("Warehouse for rent Jebel Ali", "5000 sqft warehouse in Jebel Ali Free Zone, 3 phase power, loading dock, office space included, AED 150000 per year"),
        ("Room for rent Sharjah near university", "Single room available for rent in Al Nahda Sharjah, sharing allowed, wifi included, close to University City, female only"),
        ("2 bedroom townhouse for sale", "Townhouse in Reem Island Abu Dhabi, 2 floors, 2 bedrooms, maid room, private garden, covered parking, AED 1.8 million"),
        ("Shop for rent Karama Dubai", "90 sqft retail shop in Al Karama, busy street, suitable for food kiosk or accessories, low rent AED 35000, immediate handover"),
        ("Furnished flat short term rental JBR", "Beachfront furnished apartment in JBR, 1 bedroom, daily and monthly rates available, hotel style building, steps to beach"),
        ("Labour camp accommodation Sonapur", "Labour accommodation available in Sonapur, 200 beds, canteen, security, bus pickup, suitable for construction companies"),
        ("4 bedroom villa rent Jumeirah", "Standalone villa in Jumeirah 1, 4 beds, 4 baths, private garden, maid room, driver room, corner plot, AED 250000 per year"),
        ("Retail space for rent Abu Dhabi mall", "Prime retail unit in Abu Dhabi City Centre mall, 500 sqft, high footfall location, suitable for fashion or food brand"),
        ("Studio Apartment for sale in Ajman", "Studio apartment for sale in Ajman Corniche, sea view, low floor, investors welcome, rental yield 8%, AED 180000"),
    ],

    "electronics": [
        ("iPhone 15 Pro Max 256GB for sale", "iPhone 15 Pro Max natural titanium, 256GB, purchased 2 months ago, excellent condition, full box with accessories, AED 4200"),
        ("Samsung 65 inch 4K Smart TV", "Samsung 65 inch QLED 4K smart TV, 2023 model, barely used, all cables included, pick up from Dubai Hills, AED 2800"),
        ("MacBook Pro M2 2023 16 inch", "Apple MacBook Pro 16 inch M2 Pro chip, 16GB RAM, 512GB SSD, space grey, 3 months old, original charger, AED 7500"),
        ("Sony PlayStation 5 with games", "PS5 disc edition, includes 3 games GTA FIFA and God of War, 2 controllers, excellent working condition, AED 1600"),
        ("DJI Mavic 3 Pro drone", "DJI Mavic 3 Pro drone, under 10 hours flight time, extra batteries, carry case, all accessories, no damage, AED 4500"),
        ("iPad Pro 12.9 M2 with keyboard", "iPad Pro 12.9 M2 chip, 256GB WiFi, with magic keyboard and apple pencil 2nd gen, 6 months old, AED 3800"),
        ("Canon EOS R5 mirrorless camera", "Canon EOS R5 body only, 45MP full frame, 10000 shutter actuations, excellent condition, comes with battery and charger"),
        ("Dell XPS 15 laptop 2022", "Dell XPS 15 laptop, i7 12th gen, 32GB RAM, 1TB SSD, RTX 3050 Ti, rarely used, original box, AED 4000"),
        ("Nintendo Switch OLED white", "Nintendo Switch OLED model white, with dock, 2 joy-cons, screen protector, 5 game cards included, AED 1200"),
        ("Bose QuietComfort 45 headphones", "Bose QC45 wireless noise cancelling headphones, black, 1 year old, great battery life, carrying case included, AED 750"),
        ("GoPro Hero 12 Black action camera", "GoPro Hero 12 Black, waterproof, 5.3K video, with 3 mounts, extra battery, 64GB card, used twice, AED 1100"),
        ("Samsung Galaxy S24 Ultra 512GB", "Samsung Galaxy S24 Ultra, titanium black, 512GB, S-pen included, 1 month old, full warranty remaining, AED 4800"),
        ("Apple Watch Series 9 45mm", "Apple Watch Series 9 GPS + Cellular 45mm midnight aluminium, 2 months old, extra band included, AED 1400"),
        ("LG OLED 55 inch TV", "LG C3 55 inch OLED TV, perfect blacks, game mode, comes with magic remote, wall mount included, AED 3200"),
        ("Laptop gaming ASUS ROG", "ASUS ROG Strix G15, AMD Ryzen 9, RTX 3070, 16GB RAM, 1TB SSD, 144Hz display, RGB keyboard, AED 4500"),
    ],

    "jobs": [
        ("Software Engineer needed Dubai tech startup", "We are hiring a software engineer with 3+ years experience in Python and React. Must be based in Dubai, competitive salary + visa, apply with CV"),
        ("Sales executive required real estate", "Leading real estate company in Dubai hiring experienced sales executives, commission based, training provided, own visa preferred, call to apply"),
        ("Accountant vacancy Abu Dhabi FMCG", "FMCG company in Abu Dhabi seeking qualified accountant, CPA or ACCA preferred, 5 years experience, SAP knowledge required"),
        ("Driver job available Dubai", "Private family in Jumeirah looking for experienced driver, UAE licence mandatory, must know Dubai roads well, accommodation provided"),
        ("Marketing manager position Dubai", "Digital marketing manager required for retail group, 5+ years experience, Google and Meta ads expertise, Arabic speaker preferred"),
        ("Nurse hiring Dubai private clinic", "Private medical clinic in Al Wasl hiring RN nurses, DHA licence required, 2 years UAE experience, attractive salary package"),
        ("Chef wanted 5 star hotel Abu Dhabi", "Five star hotel Abu Dhabi urgently requires experienced sous chef, international cuisine, 4 years experience, HACCP certification required"),
        ("Civil engineer vacancy UAE", "Construction company in UAE seeking civil engineer with 5+ years experience, AutoCAD proficient, site supervision role, visit visa accepted"),
        ("HR Officer job posting Sharjah", "Manufacturing company in Sharjah hiring HR Officer, 3 years experience, payroll and recruitment background, HR degree required"),
        ("Graphic designer freelance Dubai", "Creative agency in Dubai looking for talented graphic designer, Adobe Suite expert, motion graphics a plus, portfolio required, AED 6000 to 8000"),
        ("Part time teacher job Dubai", "International school hiring part time math and science teachers, teaching certificate required, flexible hours, AED 100 per hour"),
        ("Receptionist required medical center", "Medical center in Abu Dhabi hiring female receptionist, English and Arabic speaking, hospital experience preferred, immediate joining"),
        ("Data analyst wanted fintech company", "Fintech startup in DIFC hiring data analyst, Python and SQL required, tableau knowledge, 2+ years experience, AED 15000 to 20000"),
        ("Security guard job Ajman", "Security company hiring licensed security guards for mall in Ajman, SIRA licence required, shifts available, accommodation provided"),
        ("Personal trainer fitness club Dubai", "Premium fitness club in Dubai Marina seeking certified personal trainer, 2 years experience, CPR certified, excellent communication skills"),
    ],
}

def generate_dataset():
    rows = []
    for label, samples in SAMPLES.items():
        for title, desc in samples:
            text = f"{title}. {desc}"
            rows.append({"text": text, "label": label})

    # Add slight variations for more data
    extras = []
    for row in rows:
        if random.random() < 0.4:
            variation = row["text"].lower().replace(",", "").replace(".", "")
            extras.append({"text": variation, "label": row["label"]})

    all_rows = rows + extras
    random.shuffle(all_rows)
    df = pd.DataFrame(all_rows)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("data.csv", index=False)
    print(f"✅ Dataset created: {len(df)} samples")
    print(df["label"].value_counts())
