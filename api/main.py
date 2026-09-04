"""
API de detección de fraude transaccional.

Replica el pipeline de feature engineering usado en el notebook de
data_cleaning, para que una transacción "cruda" (tal como llegaría en
la vida real) pueda transformarse exactamente igual que en entrenamiento
antes de pasarla al modelo XGBoost.
"""

import json
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 1. Carga de artefactos (modelo, columnas esperadas, transformador numérico)
# ---------------------------------------------------------------------------
MODEL_PATH = "xgboost_model.pkl"
COLUMNS_PATH = "feature_columns.json"
POWER_TRANSFORMER_PATH = "power_transformer.pkl"

model = joblib.load(MODEL_PATH)
power_transformer = joblib.load(POWER_TRANSFORMER_PATH)

with open(COLUMNS_PATH, "r") as f:
    FEATURE_COLUMNS = json.load(f)

# Umbral óptimo encontrado en la validación (Prueba/OOT) para XGBoost
UMBRAL_OPTIMO = 0.89

# ---------------------------------------------------------------------------
# 2. Diccionarios de agrupación / renombrado (idénticos a data_cleaning.ipynb)
# ---------------------------------------------------------------------------
CATEGORIAS_JOB_DICT = {
    'Healthcare': [
        'Psychologist, counselling', 'Dance movement psychotherapist', 'Pathologist', 'Radiographer, diagnostic',
        'Therapist, occupational', 'Physiotherapist', 'Forensic psychologist', 'Optician, dispensing',
        'Psychologist, forensic', 'Clinical biochemist', 'Paediatric nurse', 'Child psychotherapist',
        'Paramedic', 'Audiological scientist', 'Scientist, audiological', 'Surgeon', 'Therapist, horticultural',
        'Health visitor', 'Medical secretary', 'Diagnostic radiographer', 'Medical physicist', 'Cytogeneticist',
        'Counselling psychologist', 'Chiropodist', 'Psychiatric nurse', 'Embryologist, clinical', 'Immunologist',
        'Health physicist', 'Occupational psychologist', 'Psychologist, sport and exercise', 'Doctor, hospital',
        'Phytotherapist', 'Pharmacologist', 'Horticultural therapist', 'Psychotherapist, child', 'Toxicologist',
        'Podiatrist', 'Mental health nurse', "Nurse, children's", 'Physiological scientist', 'Health and safety adviser',
        'Community pharmacist', 'Art therapist', 'Optometrist', 'Psychotherapist', 'Pharmacist, community',
        'Exercise physiologist', 'Music therapist', 'Acupuncturist', 'Hospital doctor', 'Scientist, physiological',
        'Biochemist, clinical', 'General practice doctor', 'Doctor, general practice', 'Occupational therapist',
        'Neurosurgeon', 'Orthoptist', 'Therapist, drama', 'Therapist, music', 'Dispensing optician',
        'Clinical psychologist', 'Nurse, mental health', 'Pharmacist, hospital', 'Health promotion specialist',
        'Psychiatrist', 'Radiographer, therapeutic', 'Herbalist', 'Osteopath', 'Hospital pharmacist',
        'Nutritional therapist', 'Scientist, research (medical)', 'Psychologist, clinical', 'Medical technical officer',
        'Clinical cytogeneticist', 'Homeopath', 'Veterinary surgeon',
        'Ambulance person', 'Counsellor', 'Therapist, sports', 'Clinical research associate',
        'Learning disability nurse', 'Sport and exercise psychologist', 'Research scientist (medical)', 'Oncologist'
    ],
    'Education': [
        'Special educational needs teacher', 'IT trainer', 'Education officer, museum',
        'Armed forces training and education officer', 'Higher education careers adviser',
        'English as a second language teacher', 'Administrator, education', 'Educational psychologist',
        'Teacher, English as a foreign language', 'Teacher, early years/pre', 'Primary school teacher',
        'Secondary school teacher', 'Librarian, academic', 'Further education lecturer', 'Teacher, secondary school',
        'Museum education officer', 'Teacher, special educational needs', 'Early years teacher',
        'Lecturer, further education', 'Teacher, primary school', 'Professor Emeritus', 'Community education officer',
        'Associate Professor', 'Learning mentor', 'Education administrator', 'Music tutor', 'Lecturer, higher education',
        'Teaching laboratory technician', 'English as a foreign language teacher', 'Academic librarian',
        'Teacher, adult education', 'TEFL teacher',
        'Librarian, public', 'Education officer, community', 'Careers information officer', 'Public librarian',
        'Outdoor activities/education manager', 'Environmental education officer', 'Careers adviser'
    ],
    'Tech & Engineering': [
        'Systems developer', 'Engineer, land', 'Systems analyst', 'Naval architect', 'Energy engineer',
        'Network engineer', 'Applications developer', 'Engineer, mining', 'Electrical engineer',
        'Engineer, technical sales', 'Engineer, electronics', 'Water engineer', 'Engineer, broadcasting (operations)',
        'Engineer, biomedical', 'Mining engineer', 'Engineer, communications', 'Materials engineer',
        'Engineer, structural', 'Structural engineer', 'Database administrator', 'Mechanical engineer',
        'Programmer, multimedia', 'Multimedia programmer', 'Electronics engineer', 'Chemical engineer',
        'Engineer, building services', 'Architectural technologist', 'Chief Technology Officer',
        'Control and instrumentation engineer', 'IT professional', 'Engineer, control and instrumentation',
        'Engineer, maintenance', 'Engineer, production', 'Manufacturing engineer', 'Production engineer',
        'Engineer, manufacturing', 'Engineer, drilling', 'Engineer, petroleum', 'Civil engineer, contracting',
        'Biomedical engineer', 'Building services engineer', 'Maintenance engineer', 'Site engineer',
        'Manufacturing systems engineer', 'Petroleum engineer', 'Communications engineer', 'Drilling engineer',
        'Data scientist', 'Engineer, civil (contracting)', 'IT consultant', 'Aeronautical engineer',
        'Engineer, aeronautical', 'Engineer, civil (consulting)', 'Engineer, materials', 'Broadcast engineer',
        'Engineer, site', 'Contracting civil engineer', 'Software engineer', 'Engineer, water',
        'Telecommunications researcher',
        'Architect', 'Programmer, applications', 'Engineer, agricultural', 'Engineer, automotive',
        'Statistician', 'Energy manager'
    ],
    'Business & Finance': [
        'Public affairs consultant', 'Corporate investment banker', 'Senior tax professional/tax inspector',
        'Economist', 'Purchasing manager', 'Financial adviser', 'Financial trader', 'Accounting technician',
        'Pensions consultant', 'Air broker', 'Advertising account executive', 'Advertising account planner',
        'Investment analyst', 'Pension scheme manager', 'Chief Financial Officer', 'Retail banker',
        'Sales executive', 'Insurance underwriter', 'Retail buyer', 'Equities trader', 'Risk analyst',
        'Logistics and distribution manager', 'Accountant, chartered public finance', 'Buyer, industrial',
        'Comptroller', 'Merchandiser, retail', 'Accountant, chartered certified', 'Chartered public finance accountant',
        'Chief Executive Officer', 'Chief Strategy Officer', 'Chief Operating Officer', 'Marketing executive',
        'Tax inspector', 'Chief Marketing Officer', 'Chartered accountant', 'Buyer, retail', 'Insurance broker',
        'Tax adviser', 'Management consultant', 'Investment banker, corporate', 'Company secretary', 'Media buyer',
        'Investment banker, operational', 'Industrial buyer', 'Accountant, chartered', 'Ship broker', 'Personnel officer',
        'Trade mark attorney', 'Operational researcher', 'Market researcher', 'Social researcher',
        'Social research officer, government', 'Records manager', 'Secretary/administrator', 'Public relations officer',
        'Information systems manager', 'Information officer', 'Sales professional, IT',
        'Human resources officer', 'Dealer', 'Medical sales representative', 'Training and development officer',
        'Administrator', 'Futures trader', 'Chief of Staff', 'Production manager',
        'Sales promotion account executive', 'Operational investment banker'
    ],
    'Arts & Media': [
        'Designer, multimedia', 'Programme researcher, broadcasting/film/video', 'Designer, furniture', 'Fine artist',
        'Video editor', 'Television camera operator', 'Designer, jewellery', 'Film/video editor',
        'Editor, magazine features', 'Broadcast presenter', 'Producer, radio', 'Theatre director',
        'Television production assistant', 'Exhibition designer', 'Designer, ceramics/pottery', 'Editor, film/video',
        'Camera operator', 'Copywriter, advertising', 'Designer, interior/spatial', 'Production assistant, radio',
        'Jewellery designer', 'Magazine features editor', 'Production assistant, television', 'Illustrator',
        'Designer, industrial/product', 'Writer', 'Special effects artist', 'Radio broadcast assistant',
        'Industrial/product designer', 'Ceramics designer', 'Animator', 'Arts development officer', 'Furniture designer',
        'Editor, commissioning', 'Private music teacher', 'Public relations account executive', 'Musician',
        'Therapist, art', 'Designer, exhibition/display', 'Web designer', 'Press photographer', 'Visual merchandiser',
        'Set designer', 'Television/film/video producer', 'Magazine journalist', 'Textile designer',
        'Glass blower/designer', 'Advertising copywriter', 'Artist', 'Media planner', 'Producer, television/film/video',
        'Broadcast journalist', 'Dancer', 'Designer, television/film set', 'Product designer',
        'Conservator, museum/gallery', 'Museum/gallery exhibitions officer', 'Exhibitions officer, museum/gallery',
        'Sub', 'Make', 'Copy',
        'Curator', 'Interpreter', 'Television floor manager', 'Journalist, newspaper', 'Community arts worker',
        'Radio producer', 'Commissioning editor', 'Press sub', 'Gaffer', 'Theatre manager',
        'Interior and spatial designer', 'Museum/gallery conservator', 'Presenter, broadcasting', 'Designer, textile',
        'Stage manager', 'Art gallery manager', 'Administrator, arts'
    ],
    'Trades & Manual Labor': [
        'Arboriculturist', 'Surveyor, minerals', 'Tree surgeon', 'Freight forwarder', 'Land/geomatics surveyor',
        'Building control surveyor', 'Commercial/residential surveyor', 'Mudlogger', 'Cartographer', 'Contractor',
        'Chartered loss adjuster', 'Building surveyor', 'Minerals surveyor', 'Surveyor, mining', 'Quantity surveyor',
        'Loss adjuster, chartered', 'Pilot, airline', 'Surveyor, land/geomatics', 'Quarry manager',
        'Planning and development surveyor', 'Surveyor, rural practice', 'Insurance risk surveyor',
        'Rural practice surveyor', 'Farm manager', 'Garment/textile technologist', 'Furniture conservator/restorer',
        'Surveyor, hydrographic', 'Airline pilot', 'Technical brewer', 'Land',
        'Transport planner', 'Clothing/textile technologist', 'Hydrographic surveyor', 'Conservator, furniture'
    ],
    'Public Sector & Law': [
        'Patent attorney', 'Probation officer', 'Police officer', 'Research officer, trade union',
        'Research officer, political party', 'Trading standards officer', 'Solicitor, Scotland',
        'Claims inspector/assessor', 'Historic buildings inspector/conservation officer', 'Fisheries officer',
        'Chartered legal executive (England and Wales)', 'Archivist', 'Lexicographer', 'Immigration officer',
        'Barrister', 'Administrator, local government', 'Prison officer', 'Local government officer',
        "Barrister's clerk", "Politician's assistant", 'Insurance claims handler', 'Race relations officer',
        'Advice worker', 'Warden/ranger', 'Equality and diversity officer', 'Town planner', 'Firefighter',
        'Licensed conveyancer', 'Emergency planning/management officer', 'Lawyer', 'Solicitor', 'Legal secretary',
        'Civil Service fast streamer', 'Civil Service administrator', 'Armed forces logistics/support/administrative officer',
        'Armed forces technical officer', 'Administrator, charities/voluntary organisations', 'Charity officer',
        'Charity fundraiser', 'Development worker, community', 'Aid worker',
        'Intelligence analyst', 'Regulatory affairs officer', 'Community development worker',
        'Development worker, international aid'
    ],
    'Science & Nature': [
        'Nature conservation officer', 'Geochemist', 'Scientist, research (maths)', 'Physicist, medical',
        'Amenity horticulturist', 'Science writer', 'Product/process development scientist', 'Geologist, engineering',
        'Research scientist (physical sciences)', 'Operations geologist', 'Agricultural consultant',
        'Waste management officer', 'Environmental consultant', 'Water quality scientist', 'Animal technologist',
        'Occupational hygienist', 'Landscape architect', 'Plant breeder/geneticist', 'Field seismologist',
        'Metallurgist', 'Oceanographer', 'Colour technologist', 'Geoscientist', 'Environmental health practitioner',
        'Chemist, analytical', 'Animal nutritionist', 'Soil scientist', 'Herpetologist', 'Environmental manager',
        'Horticultural consultant', 'Geophysicist/field seismologist', 'Hydrogeologist', 'Geneticist, molecular',
        'Ecologist', 'Horticulturist, commercial', 'Conservation officer, historic buildings',
        'Scientist, clinical (histocompatibility and immunogenetics)', 'Analytical chemist', 'Forest/woodland manager',
        'Engineering geologist', 'Wellsite geologist', 'Geologist, wellsite',
        'Hydrologist', 'Commercial horticulturist', 'Archaeologist', 'Scientist, marine',
        'Research scientist (life sciences)', 'Scientist, research (physical sciences)', 'Scientist, biomedical',
        'Scientific laboratory technician', 'Biomedical scientist', 'Field trials officer', 'Seismic interpreter',
        'Research scientist (maths)'
    ],
    'Service & Retail': [
        'Event organiser', 'Leisure centre manager', 'Call centre manager', 'Tourism officer',
        'Tourist information centre manager', 'Location manager', 'Health service manager', 'Retail merchandiser',
        'Bookseller', 'Facilities manager', 'Public house manager', 'Volunteer coordinator', 'Product manager',
        'Travel agency manager', 'Theme park manager', 'Heritage manager', 'Retail manager', 'Barista', 'Hotel manager',
        'Fitness centre manager', 'Estate manager/land agent', 'Catering manager', 'Warehouse manager',
        'Air cabin crew', 'Cabin crew', 'Restaurant manager, fast food', 'Tour manager', 'Customer service',
        'Air traffic controller',
        'Sports development officer', 'Sports administrator'
    ]
}

# job original (texto) -> categoría agrupada
MAPEO_JOB_EXACTO = {}
for _categoria, _lista_trabajos in CATEGORIAS_JOB_DICT.items():
    for _trabajo in _lista_trabajos:
        MAPEO_JOB_EXACTO[_trabajo] = _categoria

# Nombres de columnas dummy (inglés) -> nombres finales usados al entrenar (español)
RENAME_DICT = {
    'cat_entertainment': 'cat_entretenimiento', 'cat_food_dining': 'cat_comida_restaurantes',
    'cat_grocery_net': 'cat_supermercado_online', 'cat_grocery_pos': 'cat_supermercado_fisico',
    'cat_health_fitness': 'cat_salud_gimnasio', 'cat_home': 'cat_hogar', 'cat_kids_pets': 'cat_ninos_mascotas',
    'cat_misc_net': 'cat_miscelaneo_online', 'cat_misc_pos': 'cat_miscelaneo_fisico',
    'cat_personal_care': 'cat_cuidado_personal', 'cat_shopping_net': 'cat_compras_online',
    'cat_shopping_pos': 'cat_compras_fisico', 'cat_travel': 'cat_viajes',

    'job_Arts & Media': 'job_Artes_y_Medios', 'job_Business & Finance': 'job_Negocios_y_Finanzas',
    'job_Education': 'job_Educacion', 'job_Public Sector & Law': 'job_Sector_Publico_y_Derecho',
    'job_Science & Nature': 'job_Ciencia_y_Naturaleza', 'job_Service & Retail': 'job_Servicios_y_Comercio',
    'job_Tech & Engineering': 'job_Tecnologia_e_Ingenieria', 'job_Trades & Manual Labor': 'job_Oficios_y_Trabajo_Manual',

    'state_AK': 'estado_Alaska', 'state_AL': 'estado_Alabama', 'state_AR': 'estado_Arkansas',
    'state_AZ': 'estado_Arizona', 'state_CA': 'estado_California', 'state_CO': 'estado_Colorado',
    'state_CT': 'estado_Connecticut', 'state_DC': 'estado_Distrito_de_Columbia', 'state_DE': 'estado_Delaware',
    'state_FL': 'estado_Florida', 'state_GA': 'estado_Georgia', 'state_HI': 'estado_Hawai',
    'state_IA': 'estado_Iowa', 'state_ID': 'estado_Idaho', 'state_IL': 'estado_Illinois',
    'state_IN': 'estado_Indiana', 'state_KS': 'estado_Kansas', 'state_KY': 'estado_Kentucky',
    'state_LA': 'estado_Luisiana', 'state_MA': 'estado_Massachusetts', 'state_MD': 'estado_Maryland',
    'state_ME': 'estado_Maine', 'state_MI': 'estado_Michoacan', 'state_MN': 'estado_Minnesota',
    'state_MO': 'estado_Misuri', 'state_MS': 'estado_Misisipi', 'state_MT': 'estado_Montana',
    'state_NC': 'estado_Carolina_del_Norte', 'state_ND': 'estado_Dakota_del_Norte', 'state_NE': 'estado_Nebraska',
    'state_NH': 'estado_Nuevo_Hampshire', 'state_NJ': 'estado_Nueva_Jersey', 'state_NM': 'estado_Nuevo_Mexico',
    'state_NV': 'estado_Nevada', 'state_NY': 'estado_Nueva_York', 'state_OH': 'estado_Ohio',
    'state_OK': 'estado_Oklahoma', 'state_OR': 'estado_Oregon', 'state_PA': 'estado_Pensilvania',
    'state_RI': 'estado_Rhode_Island', 'state_SC': 'estado_Carolina_del_Sur', 'state_SD': 'estado_Dakota_del_Sur',
    'state_TN': 'estado_Tennessee', 'state_UT': 'estado_Utah', 'state_VA': 'estado_Virginia',
    'state_VT': 'estado_Vermont', 'state_WA': 'estado_Washington', 'state_WI': 'estado_Wisconsin',
    'state_WV': 'estado_Virginia_Occidental', 'state_WY': 'estado_Wyoming',
}

CATEGORIAS_VALIDAS = Literal[
    'entertainment', 'food_dining', 'gas_transport', 'grocery_net', 'grocery_pos',
    'health_fitness', 'home', 'kids_pets', 'misc_net', 'misc_pos', 'personal_care',
    'shopping_net', 'shopping_pos', 'travel'
]


# ---------------------------------------------------------------------------
# 3. Esquema de entrada (lo que recibe la API)
# ---------------------------------------------------------------------------
class Transaccion(BaseModel):
    amt: float = Field(..., gt=0, description="Monto de la transacción en USD")
    city_pop: int = Field(..., ge=0, description="Población de la ciudad del titular")
    lat: float = Field(..., description="Latitud del titular de la tarjeta")
    long: float = Field(..., description="Longitud del titular de la tarjeta")
    merch_lat: float = Field(..., description="Latitud del comercio")
    merch_long: float = Field(..., description="Longitud del comercio")
    category: CATEGORIAS_VALIDAS = Field(..., description="Categoría del comercio")
    job: str = Field(..., description="Profesión del titular (texto libre, en inglés, tal como en el dataset original)")
    state: str = Field(..., min_length=2, max_length=2, description="Código de 2 letras del estado, ej. 'CA'")

    class Config:
        json_schema_extra = {
            "example": {
                "amt": 120.50,
                "city_pop": 25000,
                "lat": 36.0,
                "long": -94.0,
                "merch_lat": 36.05,
                "merch_long": -94.10,
                "category": "shopping_net",
                "job": "Software engineer",
                "state": "CA",
            }
        }


# ---------------------------------------------------------------------------
# 4. Función que replica el pipeline de feature engineering del notebook
# ---------------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c


def construir_features(tx: Transaccion) -> pd.DataFrame:
    df = pd.DataFrame([tx.model_dump()])

    # Distancia haversine entre titular y comercio
    df["distancia_km"] = haversine(df["lat"], df["long"], df["merch_lat"], df["merch_long"])

    # Log del monto
    df["monto_log"] = np.log1p(df["amt"])

    # Agrupación de job (si no se reconoce, cae en 'NO_Mapeado' -> quedará
    # como todo-ceros en los dummies, equivalente a la categoría base 'Healthcare')
    df["job_grouped"] = df["job"].map(MAPEO_JOB_EXACTO).fillna("NO_Mapeado")

    # Escalado con el PowerTransformer ya ajustado en entrenamiento
    df[["city_pop", "distancia_km"]] = power_transformer.transform(df[["city_pop", "distancia_km"]])

    # One-hot encoding (misma lógica que pd.get_dummies en el notebook)
    df = pd.get_dummies(df, columns=["category", "job_grouped", "state"], prefix=["cat", "job", "state"], dtype=int)

    # Traducir nombres de columnas al español, igual que en entrenamiento
    df = df.rename(columns=RENAME_DICT)

    # Alinear exactamente con las columnas que el modelo espera (orden y nombre),
    # rellenando con 0 cualquier columna faltante (categorías base o no vistas)
    df_final = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    return df_final


# ---------------------------------------------------------------------------
# 5. API
# ---------------------------------------------------------------------------
app = FastAPI(
    title="API de Detección de Fraude Transaccional",
    description="Recibe los datos de una transacción y predice si es probable fraude, usando un modelo XGBoost.",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(tx: Transaccion):
    X = construir_features(tx)
    probabilidad = float(model.predict_proba(X)[0, 1])
    es_fraude = bool(probabilidad >= UMBRAL_OPTIMO)

    return {
        "probabilidad_fraude": round(probabilidad, 6),
        "es_fraude": es_fraude,
        "umbral_usado": UMBRAL_OPTIMO,
    }
