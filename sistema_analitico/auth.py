"""Autenticación simple para la demo (NO para producción).
"""
import hashlib
from sistema_analitico import db


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode('utf-8')).hexdigest()


def crear_usuario(usuario: str, password: str):
    db.crear_tabla_usuarios()
    db.agregar_usuario(usuario, hash_password(password))


def verificar_usuario(usuario: str, password: str) -> bool:
    con = db._conexion()
    cur = con.cursor()
    cur.execute('SELECT hash FROM usuarios WHERE usuario=?', (usuario,))
    row = cur.fetchone()
    con.close()
    if not row:
        return False
    return row[0] == hash_password(password)
