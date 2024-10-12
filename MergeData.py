"""
Liste détaillées des changements apportés aux données avec le préprocessing et le merge réalisés dans ce fichier.

Les features qui ont été supprimées:
    - usagers.id_usager
    - usagers.num_veh
    - usagers.id_vehicule
    - vehicules.Num_Acc
    - vehicules.num_veh
    - vehicules.occutc
    - vehicules.id_vehicule
    - lieux.voie
    - lieux.v1
    - lieux.v2
    - lieux.pr
    - lieux.pr1
    - lieux.lartpc
    - carcteristiques.an
    - carcteristiques.adr
    - carcteristiques.lat
    - carcteristiques.long
Les features qui ont été renomées:
    - usagers.an_naiss -> usagers.age
Les features qui ont été modifiées:
    - usagers.an_nais : conversion en age plutôt qu'en année de naissance
    - usagers.actp : conversion en décimal entier plutot qu'en hexadécimal
    - lieux.larrout : conversion de la largeur de la route en nombre entier représentant des centimètres plutôt qu'un flottant représentant des mètres.
    - lieux.nbv : correction du type de donnée qui été éronné dans le jeu de base.
    - carcteristiques.hrmn : approximation de l'heure à 12 minutes près et conversion en entier
    - carcteristiques.dep : conversion des numéros de département en numéros contigue
    - carcteristiques.com : conversion des numéros de commune en numéros contigue
Toutes les features ont été convertis en entiers 8bit (int8), à l'exception de:
    - Num_Acc : int64
    - com : int 16
    - larrout : int16

"""


import pandas as pd


def convert_to_centimetre(x: str) -> int:
    """
    This could've been done in a single line in the lambda function, but for the sake of readability I'm gonna write it in a separate function.
    :param x: the length you want to convert. It should be expressed in meters in a string format.
    :return: an int
    """
    x = float(x.replace(',', '.'))
    eps = 1e-4
    if -1-eps < x < -1+eps:
        return -1
    return int(x*100)

def retrieve_clean_usagers(YEAR: int) -> pd.DataFrame:
    """
    Load the usagers-YEAR.csv file and preprocess it.
      - Discard the columns "id_usager" and "num_veh"
      - Convert the brith year to an actual age and rename the column accordingly.
      - Convert the hexadecimal value of actp to an actual integer.
      - Change the type of every feature to be int8 except for the two primary keys.

    :param YEAR: The year you want to load.
    :return: a usable and preprocessed Pandas DataFrame
    """
    # Retrieve the filepath
    path = f"./data/{YEAR}/usagers-{YEAR}.csv"

    # Load the csv file as a Pandas dataframe
    usagers = pd.read_csv(path, sep=';')

    # First, drop the columns we don't need anymore
    usagers.drop(columns=["id_usager", "num_veh"], inplace=True)

    # Then take care of the an_nais column by converting the year to an age and filling the null value by -1
    usagers["an_nais"] = usagers["an_nais"].fillna(YEAR + 1).apply(lambda x: YEAR - x)
    # Properly rename this column
    usagers.rename(columns={"an_nais": "age"}, inplace=True)

    # Convert the hexadecimal value of actp to a decimal value
    usagers["actp"] = usagers["actp"].apply(lambda x: int(str(x), 16))

    # Change the type of every column to int8 except the first one.
    usagers = usagers.astype({col: 'int8' for col in usagers.columns if col not in ("id_vehicule", 'Num_Acc')})

    return usagers

def retrieve_clean_vehicules(YEAR: int) -> pd.DataFrame:
    """
    Load the vehicules-YEAR.csv file and preprocess it.
      - Discard the columns "num_veh", "occutc" and "Num_Acc"
      - Change the type of every feature to be int8 except for the primary key.

    :param YEAR: The year you want to load.
    :return: a usable and preprocessed Pandas DataFrame
    """
    # Retrieve the filepath
    path = f"./data/{YEAR}/vehicules-{YEAR}.csv"

    # Load the csv file as a Pandas dataframe
    vehicules = pd.read_csv(path, sep=';')

    # First, drop the columns we don't need anymore
    vehicules.drop(columns=["num_veh", "occutc", "Num_Acc"], inplace=True)

    # Change the type of every column to int8 except the id_vehicule.
    vehicules = vehicules.astype({col: 'int8' for col in vehicules.columns if col != "id_vehicule"})

    return vehicules

def retrieve_clean_lieux(YEAR: int) -> pd.DataFrame:
    """
    Load the lieux-YEAR.csv file and preprocess it.
      - Discard the columns "voie", "v1", "v2", "pr", "pr1" and "lartpc"
      - Convert road width to centimeters and integers
      - Correct nbv's datatype
      - Change the type of every feature to be int8 except for the primary key and larrout

    :param YEAR: The year you want to load.
    :return: a usable and preprocessed Pandas DataFrame
    """
    # Retrieve the filepath
    path = f"./data/{YEAR}/lieux-{YEAR}.csv"

    # Load the csv file as a Pandas dataframe
    lieux = pd.read_csv(path, sep=';', converters={'nbv': str})

    # Drop the unwanted features
    lieux.drop(columns=["voie", "v1", "v2", "pr", "pr1", "lartpc"], inplace=True)

    # Convert the width of the road to centimeters and integer type
    lieux["larrout"] = lieux["larrout"].apply(lambda x: convert_to_centimetre(x)).astype('int16')

    # Correct the datatype of nbv
    lieux["nbv"] = lieux["nbv"].apply(lambda x: int(x) if x != "#ERREUR" else -1)

    # Change the type of every column to int8 except the first one and larrout (because it can take values larger than 128.
    lieux = lieux.astype({col: 'int8' for col in lieux.columns if col not in ('Num_Acc', "larrout")})

    return lieux

def retrieve_clean_carcteristiques(YEAR: int) -> pd.DataFrame:
    """
    Load the carcteristiques-YEAR.csv file and preprocess it.
      - Discard the columns "an", "adr", "lat" and "long"
      - Convert and approximate time at 12 minutes.
      - Convert dep numbers to a continuous range
      - Convert com numbers to a continuous range
      - Rename Accident_Id to Num_Acc
      - Change the type of every feature to be int8 except for the primary key and com.

    :param YEAR: The year you want to load.
    :return: a usable and preprocessed Pandas DataFrame
    """
    # Retrieve the filepath
    path = f"./data/{YEAR}/carcteristiques-{YEAR}.csv"

    # Load the csv file as a Pandas dataframe
    carcteristiques = pd.read_csv(path, sep=';')

    # Drop the unwanted features
    carcteristiques.drop(columns=["an", "adr", "lat", "long"], inplace=True)

    # Let's convert time to integers approximating at 12 minutes
    carcteristiques["hrmn"] = carcteristiques["hrmn"].apply(lambda x: (int(x[:2]) * 60 + int(x[3:])) // 12)

    # Convert dep to contiguous numbers.
    mapping_departement = {val: idx for idx, val in enumerate(sorted(carcteristiques["dep"].unique()))}
    carcteristiques["dep"] = carcteristiques["dep"].map(mapping_departement)

    # Same here. Convert com to contiguous numbers.
    mapping_commune = {val: idx for idx, val in enumerate(sorted(carcteristiques["com"].unique()))}
    carcteristiques["com"] = carcteristiques["com"].map(mapping_commune).astype('int16')

    # Rename the column Accident_Id in order to match other tables
    carcteristiques.rename(columns={"Accident_Id": "Num_Acc"}, inplace=True)

    # Change the type of every column to int8 except the first one and com (because it can take values larger than 128.
    carcteristiques = carcteristiques.astype({col: 'int8' for col in carcteristiques.columns if col not in ('Num_Acc', 'com')})

    return carcteristiques

def MergeData(YEAR: int, save: bool = False) -> pd.DataFrame:
    """

    :param YEAR:
    :param save:
    :return:
    """

    merged_data = retrieve_clean_usagers(YEAR)

    merged_data = merged_data.merge(retrieve_clean_vehicules(YEAR), on="id_vehicule")
    # Now that we don't need the vehicule id anymore, we can drop it
    merged_data.drop(columns=["id_vehicule"], inplace=True)

    merged_data = merged_data.merge(retrieve_clean_lieux(YEAR), on="Num_Acc")

    merged_data = merged_data.merge(retrieve_clean_carcteristiques(YEAR), on="Num_Acc")

    if save:
        merged_data.to_pickle(f"./data/{YEAR}/merged-data-{YEAR}.pkl")

    return merged_data



