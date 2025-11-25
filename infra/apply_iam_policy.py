#!/usr/bin/env python3
"""
Script per applicare la policy IAM definita in policy.json all'utente lg-urban-modal-service.

Usage:
    python scripts/apply_iam_policy.py

Requirements:
    - boto3 installato
    - Credenziali AWS configurate (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    - Permessi IAM per modificare le policy degli utenti
"""

import json
import sys
from pathlib import Path
import boto3
from botocore.exceptions import ClientError

# Nome utente IAM target
IAM_USER_NAME = "lg-urban-modal-service"
POLICY_NAME = "S3AccessPolicy"

def load_policy_json() -> dict:
    """Carica policy.json dalla root del progetto."""
    repo_root = Path(__file__).parent.parent
    policy_file = repo_root / "policy.json"
    
    if not policy_file.exists():
        print(f"❌ Errore: {policy_file} non trovato!")
        sys.exit(1)
    
    with open(policy_file, "r") as f:
        return json.load(f)

def apply_policy():
    """Applica la policy all'utente IAM."""
    # Carica policy
    print(f"📄 Caricamento policy da policy.json...")
    policy_document = load_policy_json()
    
    # Verifica credenziali AWS
    try:
        iam = boto3.client("iam")
        # Test: verifica che le credenziali funzionino
        iam.get_user()
    except ClientError as e:
        if e.response["Error"]["Code"] == "InvalidClientTokenId":
            print("❌ Errore: Credenziali AWS non valide o non configurate!")
            print("   Configura AWS_ACCESS_KEY_ID e AWS_SECRET_ACCESS_KEY")
            sys.exit(1)
        raise
    
    # Verifica che l'utente esista
    try:
        iam.get_user(UserName=IAM_USER_NAME)
        print(f"✅ Utente IAM trovato: {IAM_USER_NAME}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchEntity":
            print(f"❌ Errore: Utente IAM '{IAM_USER_NAME}' non trovato!")
            sys.exit(1)
        raise
    
    # Applica la policy (put-user-policy crea o aggiorna una policy inline)
    try:
        print(f"🔧 Applicazione policy '{POLICY_NAME}' all'utente {IAM_USER_NAME}...")
        iam.put_user_policy(
            UserName=IAM_USER_NAME,
            PolicyName=POLICY_NAME,
            PolicyDocument=json.dumps(policy_document)
        )
        print(f"✅ Policy applicata con successo!")
        
        # Verifica che sia stata applicata
        response = iam.get_user_policy(
            UserName=IAM_USER_NAME,
            PolicyName=POLICY_NAME
        )
        print(f"\n📋 Policy verificata:")
        print(f"   Nome: {response['PolicyName']}")
        print(f"   Utente: {response['UserName']}")
        
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "AccessDenied":
            print(f"❌ Errore: Permessi insufficienti per modificare la policy!")
            print(f"   Assicurati di avere i permessi iam:PutUserPolicy")
            sys.exit(1)
        else:
            print(f"❌ Errore durante l'applicazione della policy: {e}")
            sys.exit(1)

if __name__ == "__main__":
    print("🚀 Applicazione policy IAM\n")
    apply_policy()
    print("\n✨ Completato!")