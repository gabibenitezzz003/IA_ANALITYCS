"""Módulo simple de persistencia con SQLite para datasets y usuarios.
"""
import sqlite3
import pandas as pd
from typing import Optional
import os

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'sistema.db')


def _conexion(path: Optional[str] = None):
    path = path or DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return sqlite3.connect(path)


def guardar_dataframe(nombre: str, df: pd.DataFrame):
    with _conexion() as con:
        df.to_sql(nombre, con, if_exists='replace', index=False)


def cargar_dataframe(nombre: str) -> pd.DataFrame:
    with _conexion() as con:
        return pd.read_sql_query(f"SELECT * FROM '{nombre}'", con)


def crear_tabla_usuarios():
    with _conexion() as con:
        cur = con.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS usuarios (id INTEGER PRIMARY KEY, usuario TEXT UNIQUE, hash TEXT)''')
        con.commit()


def agregar_usuario(usuario: str, hash_pw: str):
    with _conexion() as con:
        cur = con.cursor()
        cur.execute('INSERT OR REPLACE INTO usuarios (usuario, hash) VALUES (?, ?)', (usuario, hash_pw))
        con.commit()
