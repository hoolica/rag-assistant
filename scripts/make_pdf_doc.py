"""Génère le document PDF de démonstration (contrat de gym fictif), pour montrer l'ingestion de PDF."""

from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parents[1] / "data" / "docs" / "abonnement_gym.pdf"

SECTIONS = [
    ("Contrat d'abonnement — Centre de conditionnement Énergie+ (document fictif)", None),
    ("Membre et durée", "Membre : Alex Tremblay. Abonnement mensuel sans terme, débuté le 1er septembre 2026, "
     "renouvelé automatiquement chaque mois."),
    ("Tarif", "Le tarif est de 39,99 $ par mois, taxes en sus, prélevé le 1er de chaque mois. "
     "Des frais d'inscription de 49 $ ont été payés à l'adhésion et ne sont pas remboursables."),
    ("Annulation", "Le membre peut annuler son abonnement en tout temps en remplissant le formulaire "
     "d'annulation à l'accueil ou par courriel, avec un préavis de 30 jours. Le dernier prélèvement a lieu "
     "pendant cette période de préavis."),
    ("Gel de l'abonnement", "L'abonnement peut être gelé (suspendu) pour une durée maximale de 3 mois par année, "
     "sur présentation d'un billet médical ou pour un séjour à l'étranger. Aucun frais n'est prélevé pendant le gel."),
    ("Accès", "L'accès est inclus 7 jours sur 7, de 5 h à 23 h. Les cours de groupe sont inclus ; "
     "l'entraînement privé est en supplément."),
]


def main():
    styles = getSampleStyleSheet()
    story = []
    for title, body in SECTIONS:
        story.append(Paragraph(title, styles["Title" if body is None else "Heading2"]))
        if body:
            story.append(Paragraph(body, styles["BodyText"]))
        story.append(Spacer(1, 8))
    SimpleDocTemplate(str(OUT), pagesize=LETTER).build(story)
    print(f"PDF généré : {OUT}")


if __name__ == "__main__":
    main()
