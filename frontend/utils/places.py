"""
Turns grid-cell coordinates into place descriptions people recognise:
-33.5, 151.0  ->  "near Gosford, NSW"  or  "40 km NW of Katherine, NT".

The town list is kept here as Python rather than a CSV on purpose: every
data/*.csv is stored in Git LFS (see .gitattributes), and without git-lfs
installed a CSV would arrive as a 3-line pointer file and break this lookup.

Coordinates are approximate town centres (to about 0.05 degrees), compiled
for this dashboard. That is far finer than the 0.5-degree (~55 km) forecast
grid, so it is accurate enough to say which town a cell is near -- it is not
a gazetteer. The list deliberately includes small remote towns, so outback
cells get a nearby reference point instead of a capital city 800 km away.
"""

from functools import lru_cache

import numpy as np

EARTH_RADIUS_KM = 6371.0
NEAR_KM = 25  # closer than this reads as "near <town>", no distance needed

# (name, state, latitude, longitude)
TOWNS = [
    # --- New South Wales ---
    ("Sydney", "NSW", -33.87, 151.21), ("Newcastle", "NSW", -32.93, 151.78),
    ("Wollongong", "NSW", -34.42, 150.89), ("Gosford", "NSW", -33.43, 151.34),
    ("Port Macquarie", "NSW", -31.43, 152.91), ("Coffs Harbour", "NSW", -30.30, 153.11),
    ("Grafton", "NSW", -29.69, 152.93), ("Lismore", "NSW", -28.81, 153.28),
    ("Tweed Heads", "NSW", -28.18, 153.54), ("Casino", "NSW", -28.87, 153.05),
    ("Armidale", "NSW", -30.51, 151.67), ("Tamworth", "NSW", -31.09, 150.93),
    ("Moree", "NSW", -29.47, 149.84), ("Narrabri", "NSW", -30.32, 149.78),
    ("Inverell", "NSW", -29.78, 151.11), ("Glen Innes", "NSW", -29.74, 151.74),
    ("Tenterfield", "NSW", -29.05, 152.02), ("Gunnedah", "NSW", -30.98, 150.25),
    ("Coonabarabran", "NSW", -31.27, 149.28), ("Dubbo", "NSW", -32.25, 148.60),
    ("Orange", "NSW", -33.28, 149.10), ("Bathurst", "NSW", -33.42, 149.58),
    ("Lithgow", "NSW", -33.48, 150.16), ("Katoomba", "NSW", -33.71, 150.31),
    ("Mudgee", "NSW", -32.59, 149.59), ("Muswellbrook", "NSW", -32.26, 150.89),
    ("Taree", "NSW", -31.90, 152.46), ("Kempsey", "NSW", -31.08, 152.84),
    ("Goulburn", "NSW", -34.75, 149.72), ("Yass", "NSW", -34.84, 148.91),
    ("Young", "NSW", -34.31, 148.30), ("Cowra", "NSW", -33.83, 148.69),
    ("Parkes", "NSW", -33.14, 148.17), ("Forbes", "NSW", -33.38, 148.01),
    ("Condobolin", "NSW", -33.09, 147.15), ("West Wyalong", "NSW", -33.92, 147.21),
    ("Wagga Wagga", "NSW", -35.12, 147.37), ("Tumut", "NSW", -35.30, 148.22),
    ("Albury", "NSW", -36.08, 146.92), ("Griffith", "NSW", -34.29, 146.04),
    ("Leeton", "NSW", -34.55, 146.40), ("Hillston", "NSW", -33.48, 145.54),
    ("Hay", "NSW", -34.51, 144.85), ("Deniliquin", "NSW", -35.53, 144.96),
    ("Balranald", "NSW", -34.64, 143.56), ("Wentworth", "NSW", -34.11, 141.92),
    ("Cooma", "NSW", -36.24, 149.13), ("Jindabyne", "NSW", -36.42, 148.62),
    ("Bega", "NSW", -36.67, 149.84), ("Eden", "NSW", -37.06, 149.90),
    ("Batemans Bay", "NSW", -35.71, 150.18), ("Nowra", "NSW", -34.88, 150.60),
    ("Nyngan", "NSW", -31.56, 147.19), ("Cobar", "NSW", -31.50, 145.84),
    ("Bourke", "NSW", -30.09, 145.94), ("Brewarrina", "NSW", -29.96, 146.86),
    ("Walgett", "NSW", -30.02, 148.12), ("Lightning Ridge", "NSW", -29.43, 147.98),
    ("Wilcannia", "NSW", -31.56, 143.38), ("Ivanhoe", "NSW", -32.90, 144.30),
    ("White Cliffs", "NSW", -30.85, 143.09), ("Broken Hill", "NSW", -31.95, 141.47),
    ("Tibooburra", "NSW", -29.43, 142.01),
    # --- Australian Capital Territory ---
    ("Canberra", "ACT", -35.28, 149.13),
    # --- Victoria ---
    ("Melbourne", "VIC", -37.81, 144.96), ("Geelong", "VIC", -38.15, 144.36),
    ("Ballarat", "VIC", -37.56, 143.85), ("Bendigo", "VIC", -36.76, 144.28),
    ("Shepparton", "VIC", -36.38, 145.40), ("Wodonga", "VIC", -36.12, 146.89),
    ("Wangaratta", "VIC", -36.36, 146.31), ("Benalla", "VIC", -36.55, 145.98),
    ("Seymour", "VIC", -37.03, 145.14), ("Mansfield", "VIC", -37.05, 146.08),
    ("Bright", "VIC", -36.73, 146.96), ("Omeo", "VIC", -37.10, 147.60),
    ("Healesville", "VIC", -37.65, 145.52), ("Castlemaine", "VIC", -37.06, 144.22),
    ("Echuca", "VIC", -36.13, 144.75), ("Kerang", "VIC", -35.73, 143.92),
    ("Swan Hill", "VIC", -35.34, 143.55), ("Mildura", "VIC", -34.19, 142.16),
    ("Ouyen", "VIC", -35.07, 142.32), ("St Arnaud", "VIC", -36.62, 143.26),
    ("Horsham", "VIC", -36.71, 142.20), ("Nhill", "VIC", -36.33, 141.65),
    ("Stawell", "VIC", -37.06, 142.78), ("Halls Gap", "VIC", -37.14, 142.52),
    ("Ararat", "VIC", -37.28, 142.93), ("Hamilton", "VIC", -37.74, 142.02),
    ("Portland", "VIC", -38.34, 141.60), ("Warrnambool", "VIC", -38.38, 142.48),
    ("Colac", "VIC", -38.34, 143.58), ("Wonthaggi", "VIC", -38.61, 145.59),
    ("Warragul", "VIC", -38.16, 145.93), ("Leongatha", "VIC", -38.48, 145.95),
    ("Traralgon", "VIC", -38.20, 146.54), ("Sale", "VIC", -38.11, 147.07),
    ("Bairnsdale", "VIC", -37.83, 147.61), ("Lakes Entrance", "VIC", -37.88, 147.98),
    ("Orbost", "VIC", -37.71, 148.46), ("Mallacoota", "VIC", -37.56, 149.75),
    # --- Queensland ---
    ("Brisbane", "QLD", -27.47, 153.03), ("Gold Coast", "QLD", -28.02, 153.40),
    ("Maroochydore", "QLD", -26.65, 153.09), ("Ipswich", "QLD", -27.61, 152.76),
    ("Toowoomba", "QLD", -27.56, 151.95), ("Warwick", "QLD", -28.22, 152.03),
    ("Stanthorpe", "QLD", -28.65, 151.93), ("Goondiwindi", "QLD", -28.55, 150.31),
    ("St George", "QLD", -28.04, 148.58), ("Dalby", "QLD", -27.18, 151.26),
    ("Chinchilla", "QLD", -26.74, 150.63), ("Kingaroy", "QLD", -26.54, 151.84),
    ("Roma", "QLD", -26.57, 148.79), ("Injune", "QLD", -25.84, 148.57),
    ("Mitchell", "QLD", -26.49, 147.98), ("Charleville", "QLD", -26.40, 146.24),
    ("Augathella", "QLD", -25.80, 146.58), ("Cunnamulla", "QLD", -28.07, 145.68),
    ("Thargomindah", "QLD", -27.99, 143.82), ("Quilpie", "QLD", -26.61, 144.27),
    ("Windorah", "QLD", -25.42, 142.66), ("Birdsville", "QLD", -25.90, 139.35),
    ("Bedourie", "QLD", -24.36, 139.47), ("Boulia", "QLD", -22.91, 139.91),
    ("Gympie", "QLD", -26.19, 152.67), ("Maryborough", "QLD", -25.54, 152.70),
    ("Hervey Bay", "QLD", -25.29, 152.85), ("Bundaberg", "QLD", -24.87, 152.35),
    ("Gladstone", "QLD", -23.84, 151.26), ("Biloela", "QLD", -24.40, 150.51),
    ("Rockhampton", "QLD", -23.38, 150.51), ("Yeppoon", "QLD", -23.13, 150.74),
    ("Emerald", "QLD", -23.53, 148.16), ("Springsure", "QLD", -24.12, 148.09),
    ("Clermont", "QLD", -22.82, 147.64), ("Moranbah", "QLD", -22.00, 148.05),
    ("Mackay", "QLD", -21.14, 149.19), ("Proserpine", "QLD", -20.40, 148.58),
    ("Bowen", "QLD", -20.01, 148.25), ("Townsville", "QLD", -19.26, 146.82),
    ("Charters Towers", "QLD", -20.08, 146.26), ("Ingham", "QLD", -18.65, 146.16),
    ("Innisfail", "QLD", -17.52, 146.03), ("Cairns", "QLD", -16.92, 145.77),
    ("Atherton", "QLD", -17.27, 145.48), ("Port Douglas", "QLD", -16.48, 145.47),
    ("Cooktown", "QLD", -15.47, 145.25), ("Laura", "QLD", -15.56, 144.45),
    ("Coen", "QLD", -13.95, 143.20), ("Lockhart River", "QLD", -12.79, 143.34),
    ("Aurukun", "QLD", -13.36, 141.73), ("Weipa", "QLD", -12.63, 141.88),
    ("Bamaga", "QLD", -10.89, 142.39), ("Pormpuraaw", "QLD", -14.90, 141.62),
    ("Kowanyama", "QLD", -15.48, 141.75), ("Karumba", "QLD", -17.49, 140.84),
    ("Normanton", "QLD", -17.67, 141.08), ("Croydon", "QLD", -18.20, 142.24),
    ("Georgetown", "QLD", -18.29, 143.55), ("Burketown", "QLD", -17.74, 139.55),
    ("Doomadgee", "QLD", -17.94, 138.82), ("Gununa", "QLD", -16.66, 139.18),
    ("Hughenden", "QLD", -20.84, 144.20), ("Richmond", "QLD", -20.73, 143.14),
    ("Julia Creek", "QLD", -20.66, 141.75), ("Cloncurry", "QLD", -20.71, 140.51),
    ("Mount Isa", "QLD", -20.73, 139.49), ("Camooweal", "QLD", -19.92, 138.12),
    ("Winton", "QLD", -22.39, 143.04), ("Muttaburra", "QLD", -22.59, 144.55),
    ("Aramac", "QLD", -22.97, 145.24), ("Longreach", "QLD", -23.44, 144.25),
    ("Barcaldine", "QLD", -23.55, 145.29), ("Isisford", "QLD", -24.26, 144.44),
    ("Blackall", "QLD", -24.42, 145.47), ("Tambo", "QLD", -24.88, 146.26),
    # --- Northern Territory ---
    ("Darwin", "NT", -12.46, 130.84), ("Palmerston", "NT", -12.49, 130.98),
    ("Wurrumiyanga", "NT", -11.76, 130.63), ("Batchelor", "NT", -13.05, 131.03),
    ("Adelaide River", "NT", -13.24, 131.10), ("Pine Creek", "NT", -13.82, 131.83),
    ("Jabiru", "NT", -12.67, 132.84), ("Gunbalanya", "NT", -12.33, 133.05),
    ("Maningrida", "NT", -12.06, 134.23), ("Galiwinku", "NT", -12.03, 135.57),
    ("Nhulunbuy", "NT", -12.18, 136.78), ("Numbulwar", "NT", -14.27, 135.73),
    ("Ngukurr", "NT", -14.73, 134.73), ("Borroloola", "NT", -16.07, 136.31),
    ("Katherine", "NT", -14.47, 132.26), ("Barunga", "NT", -14.52, 132.87),
    ("Mataranka", "NT", -14.92, 133.07), ("Wadeye", "NT", -14.24, 129.52),
    ("Timber Creek", "NT", -15.65, 130.48), ("Daly Waters", "NT", -16.25, 133.37),
    ("Elliott", "NT", -17.55, 133.54), ("Kalkarindji", "NT", -17.45, 130.83),
    ("Lajamanu", "NT", -18.33, 130.64), ("Tennant Creek", "NT", -19.65, 134.19),
    ("Ali Curung", "NT", -21.02, 134.33), ("Alpurrurulam", "NT", -20.98, 137.84),
    ("Ti Tree", "NT", -22.13, 133.42), ("Yuendumu", "NT", -22.25, 131.80),
    ("Papunya", "NT", -23.21, 131.91), ("Kintore", "NT", -23.27, 129.39),
    ("Alice Springs", "NT", -23.70, 133.88), ("Hermannsburg", "NT", -23.94, 132.78),
    ("Santa Teresa", "NT", -24.13, 134.37), ("Yulara", "NT", -25.24, 130.99),
    ("Docker River", "NT", -24.86, 129.09), ("Kulgera", "NT", -25.84, 133.30),
    ("Finke", "NT", -25.59, 134.58),
    # --- Western Australia ---
    ("Perth", "WA", -31.95, 115.86), ("Mandurah", "WA", -32.53, 115.72),
    ("Bunbury", "WA", -33.33, 115.64), ("Busselton", "WA", -33.65, 115.35),
    ("Margaret River", "WA", -33.95, 115.07), ("Augusta", "WA", -34.31, 115.16),
    ("Collie", "WA", -33.36, 116.16), ("Bridgetown", "WA", -33.96, 116.14),
    ("Manjimup", "WA", -34.24, 116.15), ("Pemberton", "WA", -34.44, 116.03),
    ("Walpole", "WA", -34.98, 116.73), ("Denmark", "WA", -34.96, 117.35),
    ("Albany", "WA", -35.02, 117.88), ("Mount Barker", "WA", -34.63, 117.67),
    ("Kojonup", "WA", -33.83, 117.16), ("Katanning", "WA", -33.69, 117.56),
    ("Wagin", "WA", -33.31, 117.34), ("Narrogin", "WA", -32.93, 117.18),
    ("Lake Grace", "WA", -33.10, 118.46), ("Kulin", "WA", -32.67, 118.16),
    ("Hyden", "WA", -32.45, 118.86), ("Jerramungup", "WA", -33.94, 118.92),
    ("Bremer Bay", "WA", -34.40, 119.38), ("Ravensthorpe", "WA", -33.58, 120.05),
    ("Hopetoun", "WA", -33.95, 120.13), ("Esperance", "WA", -33.86, 121.89),
    ("Norseman", "WA", -32.20, 121.78), ("Balladonia", "WA", -32.35, 123.62),
    ("Cocklebiddy", "WA", -32.04, 125.91), ("Eucla", "WA", -31.68, 128.88),
    ("Kambalda", "WA", -31.20, 121.67), ("Coolgardie", "WA", -30.95, 121.16),
    ("Kalgoorlie", "WA", -30.75, 121.47), ("Menzies", "WA", -29.69, 121.03),
    ("Leonora", "WA", -28.88, 121.33), ("Laverton", "WA", -28.63, 122.40),
    ("Leinster", "WA", -27.92, 120.70), ("Wiluna", "WA", -26.59, 120.23),
    ("Warburton", "WA", -26.13, 126.58), ("Southern Cross", "WA", -31.23, 119.33),
    ("Merredin", "WA", -31.48, 118.28), ("Narembeen", "WA", -32.06, 118.39),
    ("Bruce Rock", "WA", -31.88, 118.15), ("Kellerberrin", "WA", -31.63, 117.72),
    ("Mukinbudin", "WA", -30.92, 118.21), ("Beverley", "WA", -32.11, 116.92),
    ("York", "WA", -31.89, 116.77), ("Northam", "WA", -31.65, 116.67),
    ("Toodyay", "WA", -31.55, 116.47), ("Gingin", "WA", -31.35, 115.90),
    ("Wongan Hills", "WA", -30.89, 116.72), ("Dalwallinu", "WA", -30.28, 116.66),
    ("Moora", "WA", -30.64, 116.01), ("Lancelin", "WA", -31.02, 115.33),
    ("Cervantes", "WA", -30.50, 115.06), ("Jurien Bay", "WA", -30.30, 115.04),
    ("Three Springs", "WA", -29.53, 115.76), ("Morawa", "WA", -29.21, 116.01),
    ("Dongara", "WA", -29.25, 114.93), ("Geraldton", "WA", -28.78, 114.61),
    ("Mullewa", "WA", -28.54, 115.51), ("Northampton", "WA", -28.35, 114.64),
    ("Yalgoo", "WA", -28.34, 116.68), ("Kalbarri", "WA", -27.71, 114.17),
    ("Mount Magnet", "WA", -28.06, 117.85), ("Sandstone", "WA", -27.99, 119.30),
    ("Cue", "WA", -27.42, 117.90), ("Meekatharra", "WA", -26.59, 118.50),
    ("Denham", "WA", -25.93, 113.53), ("Carnarvon", "WA", -24.88, 113.66),
    ("Gascoyne Junction", "WA", -25.05, 115.21), ("Coral Bay", "WA", -23.14, 113.77),
    ("Exmouth", "WA", -21.93, 114.13), ("Onslow", "WA", -21.64, 115.11),
    ("Pannawonica", "WA", -21.64, 116.32), ("Paraburdoo", "WA", -23.20, 117.67),
    ("Tom Price", "WA", -22.69, 117.79), ("Newman", "WA", -23.36, 119.73),
    ("Dampier", "WA", -20.66, 116.71), ("Karratha", "WA", -20.74, 116.85),
    ("Roebourne", "WA", -20.78, 117.15), ("Port Hedland", "WA", -20.31, 118.58),
    ("Marble Bar", "WA", -21.18, 119.75), ("Nullagine", "WA", -21.89, 120.11),
    ("Broome", "WA", -17.96, 122.24), ("Derby", "WA", -17.30, 123.63),
    ("Fitzroy Crossing", "WA", -18.19, 125.58), ("Halls Creek", "WA", -18.22, 127.67),
    ("Balgo", "WA", -20.14, 127.98), ("Wyndham", "WA", -15.49, 128.12),
    ("Kununurra", "WA", -15.77, 128.74), ("Kalumburu", "WA", -14.30, 126.64),
    # --- South Australia ---
    ("Adelaide", "SA", -34.93, 138.60), ("Gawler", "SA", -34.60, 138.75),
    ("Mount Barker", "SA", -35.07, 138.86), ("Victor Harbor", "SA", -35.55, 138.62),
    ("Murray Bridge", "SA", -35.12, 139.27), ("Mannum", "SA", -34.91, 139.30),
    ("Tailem Bend", "SA", -35.25, 139.45), ("Meningie", "SA", -35.69, 139.34),
    ("Kingscote", "SA", -35.66, 137.64), ("Parndana", "SA", -35.79, 137.26),
    ("Kingston SE", "SA", -36.83, 139.85), ("Robe", "SA", -37.16, 139.76),
    ("Millicent", "SA", -37.60, 140.35), ("Mount Gambier", "SA", -37.83, 140.78),
    ("Naracoorte", "SA", -36.96, 140.74), ("Bordertown", "SA", -36.31, 140.77),
    ("Keith", "SA", -36.10, 140.35), ("Lameroo", "SA", -35.33, 140.52),
    ("Pinnaroo", "SA", -35.26, 140.91), ("Loxton", "SA", -34.45, 140.57),
    ("Berri", "SA", -34.28, 140.60), ("Renmark", "SA", -34.18, 140.75),
    ("Waikerie", "SA", -34.18, 139.98), ("Morgan", "SA", -34.03, 139.67),
    ("Clare", "SA", -33.83, 138.61), ("Burra", "SA", -33.68, 138.94),
    ("Jamestown", "SA", -33.20, 138.60), ("Peterborough", "SA", -32.97, 138.84),
    ("Orroroo", "SA", -32.73, 138.61), ("Port Pirie", "SA", -33.19, 138.02),
    ("Port Broughton", "SA", -33.60, 137.93), ("Port Wakefield", "SA", -34.19, 138.15),
    ("Kadina", "SA", -33.96, 137.72), ("Minlaton", "SA", -34.77, 137.60),
    ("Yorketown", "SA", -35.02, 137.61), ("Port Augusta", "SA", -32.49, 137.77),
    ("Quorn", "SA", -32.35, 138.04), ("Hawker", "SA", -31.89, 138.42),
    ("Whyalla", "SA", -33.03, 137.58), ("Cowell", "SA", -33.68, 136.92),
    ("Kimba", "SA", -33.14, 136.42), ("Tumby Bay", "SA", -34.38, 136.10),
    ("Port Lincoln", "SA", -34.73, 135.86), ("Cummins", "SA", -34.26, 135.73),
    ("Wudinna", "SA", -33.05, 135.46), ("Streaky Bay", "SA", -32.80, 134.21),
    ("Ceduna", "SA", -32.13, 133.68), ("Penong", "SA", -31.93, 133.01),
    ("Yalata", "SA", -31.48, 131.84), ("Nullarbor Roadhouse", "SA", -31.45, 130.90),
    ("Tarcoola", "SA", -30.71, 134.57), ("Glendambo", "SA", -30.97, 135.75),
    ("Woomera", "SA", -31.20, 136.83), ("Roxby Downs", "SA", -30.56, 136.90),
    ("Andamooka", "SA", -30.45, 137.17), ("Leigh Creek", "SA", -30.59, 138.41),
    ("Arkaroola", "SA", -30.31, 139.34), ("Marree", "SA", -29.65, 138.06),
    ("Innamincka", "SA", -27.75, 140.73), ("Coober Pedy", "SA", -29.01, 134.75),
    ("Oodnadatta", "SA", -27.55, 135.45), ("Marla", "SA", -27.30, 133.62),
    ("Indulkana", "SA", -26.97, 133.31), ("Pukatja", "SA", -26.27, 132.14),
    ("Amata", "SA", -26.15, 131.15),
    # --- Tasmania ---
    ("Hobart", "TAS", -42.88, 147.33), ("Launceston", "TAS", -41.44, 147.14),
    ("Devonport", "TAS", -41.18, 146.35), ("Burnie", "TAS", -41.05, 145.91),
    ("Wynyard", "TAS", -40.99, 145.73), ("Stanley", "TAS", -40.76, 145.30),
    ("Smithton", "TAS", -40.84, 145.12), ("George Town", "TAS", -41.10, 146.83),
    ("Scottsdale", "TAS", -41.16, 147.52), ("St Helens", "TAS", -41.32, 148.25),
    ("Bicheno", "TAS", -41.87, 148.30), ("Swansea", "TAS", -42.12, 148.07),
    ("Triabunna", "TAS", -42.51, 147.91), ("Sorell", "TAS", -42.78, 147.56),
    ("Nubeena", "TAS", -43.10, 147.74), ("Huonville", "TAS", -43.03, 147.05),
    ("Dover", "TAS", -43.31, 147.01), ("New Norfolk", "TAS", -42.78, 147.06),
    ("Maydena", "TAS", -42.76, 146.62), ("Bothwell", "TAS", -42.38, 147.01),
    ("Oatlands", "TAS", -42.30, 147.37), ("Campbell Town", "TAS", -41.93, 147.49),
    ("Longford", "TAS", -41.60, 147.12), ("Deloraine", "TAS", -41.52, 146.66),
    ("Sheffield", "TAS", -41.38, 146.33), ("Miena", "TAS", -41.98, 146.73),
    ("Tullah", "TAS", -41.74, 145.61), ("Rosebery", "TAS", -41.78, 145.54),
    ("Zeehan", "TAS", -41.88, 145.34), ("Queenstown", "TAS", -42.08, 145.56),
    ("Strahan", "TAS", -42.15, 145.33), ("Whitemark", "TAS", -40.12, 148.02),
    ("Currie", "TAS", -39.93, 143.85),
]

_NAMES = [f"{name}, {state}" for name, state, _, _ in TOWNS]
_LAT = np.radians([t[2] for t in TOWNS])
_LON = np.radians([t[3] for t in TOWNS])
_COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def _distances_km(lat: float, lon: float) -> np.ndarray:
    """Great-circle distance from (lat, lon) to every town (haversine)."""
    la, lo = np.radians(lat), np.radians(lon)
    a = (np.sin((_LAT - la) / 2) ** 2
         + np.cos(la) * np.cos(_LAT) * np.sin((_LON - lo) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _bearing(from_lat: float, from_lon: float, to_lat: float, to_lon: float) -> str:
    """8-point compass direction from the town to the cell."""
    la1, la2 = np.radians(from_lat), np.radians(to_lat)
    dlon = np.radians(to_lon - from_lon)
    y = np.sin(dlon) * np.cos(la2)
    x = np.cos(la1) * np.sin(la2) - np.sin(la1) * np.cos(la2) * np.cos(dlon)
    degrees = (np.degrees(np.arctan2(y, x)) + 360) % 360
    return _COMPASS[int((degrees + 22.5) // 45) % 8]


@lru_cache(maxsize=8192)
def nearest_town(lat: float, lon: float) -> tuple:
    """Returns ("Katherine, NT", distance_km, compass direction from the town)."""
    dist = _distances_km(lat, lon)
    i = int(np.argmin(dist))
    return _NAMES[i], float(dist[i]), _bearing(TOWNS[i][2], TOWNS[i][3], lat, lon)


def describe_location(lat: float, lon: float) -> str:
    """
    "near Katherine, NT" when the cell is close to a town, otherwise
    "40 km NW of Katherine, NT". Distances are rounded to 5 km -- the cell
    is ~55 km across, so more precision would be false precision.
    """
    name, km, direction = nearest_town(float(lat), float(lon))
    if km < NEAR_KM:
        return f"near {name}"
    return f"{int(round(km / 5) * 5)} km {direction} of {name}"
