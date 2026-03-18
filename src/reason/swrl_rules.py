"""
swrl_rules.py
-------------
Raisonnement SWRL avec OWLReady2.

Partie 1 : Règles sur family.owl (exercice du cours)
Partie 2 : Règles sur le KB des scientifiques

Usage (depuis la racine du projet) :
    python src/reason/swrl_rules.py

Dépendances :
    pip install owlready2 rdflib
"""

from owlready2 import (
    get_ontology, Thing, ObjectProperty,
    AllDisjoint, Imp
)


# ══════════════════════════════════════════════════════════════
#  PARTIE 1 — family.owl
#  Règle : Person(?p) ∧ hasSibling(?p,?s) ∧ Man(?s) → hasBrother(?p,?s)
# ══════════════════════════════════════════════════════════════

def run_family_rules():
    print("\n" + "═"*60)
    print("PARTIE 1 — Règles SWRL sur family.owl")
    print("═"*60)

    onto = get_ontology("http://example.org/family.owl#")

    with onto:
        class Person(Thing): pass
        class Man(Person): pass
        class Woman(Person): pass
        AllDisjoint([Man, Woman])

        class hasSibling(ObjectProperty):
            domain    = [Person]
            range     = [Person]
            symmetric = True

        class hasBrother(ObjectProperty):
            domain = [Person]
            range  = [Person]

        class hasParent(ObjectProperty):
            domain = [Person]
            range  = [Person]

        class hasGrandParent(ObjectProperty):
            domain = [Person]
            range  = [Person]

        alice = Person("Alice")
        bob   = Man("Bob")
        carol = Person("Carol")
        david = Man("David")
        eve   = Woman("Eve")

        alice.hasSibling = [bob, carol]
        carol.hasSibling = [alice, bob]
        bob.hasSibling   = [alice, carol]

        carol.hasParent  = [david]
        david.hasParent  = [eve]

        rule1 = Imp()
        rule1.set_as_rule(
            "Person(?p), hasSibling(?p, ?s), Man(?s) -> hasBrother(?p, ?s)"
        )

        rule2 = Imp()
        rule2.set_as_rule(
            "Person(?p), hasParent(?p, ?q), hasParent(?q, ?g) -> hasGrandParent(?p, ?g)"
        )

    print("\nApplication manuelle des règles SWRL :\n")

    print("Règle 1 : Person(?p) ∧ hasSibling(?p,?s) ∧ Man(?s) → hasBrother(?p,?s)")
    results_r1 = []
    for p in onto.individuals():
        for s in getattr(p, "hasSibling", []):
            if isinstance(s, Man):
                current = list(getattr(p, "hasBrother", []))
                if s not in current:
                    current.append(s)
                    p.hasBrother = current
                results_r1.append((p.name, s.name))
                print(f"  → hasBrother({p.name}, {s.name})")

    if not results_r1:
        print("  (aucun résultat)")

    print("\nRègle 2 : Person(?p) ∧ hasParent(?p,?q) ∧ hasParent(?q,?g) → hasGrandParent(?p,?g)")
    results_r2 = []
    for p in onto.individuals():
        for q in getattr(p, "hasParent", []):
            for g in getattr(q, "hasParent", []):
                if p != g:
                    current = list(getattr(p, "hasGrandParent", []))
                    if g not in current:
                        current.append(g)
                        p.hasGrandParent = current
                    results_r2.append((p.name, g.name))
                    print(f"  → hasGrandParent({p.name}, {g.name})")

    if not results_r2:
        print("  (aucun résultat)")

    print(f"\nRésumé famille :")
    print(f"  hasBrother inférés     : {len(results_r1)}")
    print(f"  hasGrandParent inférés : {len(results_r2)}")


# ══════════════════════════════════════════════════════════════
#  PARTIE 2 — KB des scientifiques
# ══════════════════════════════════════════════════════════════

def run_scientist_rules(kb_path="kg_artifacts/expanded.nt"):
    print("\n" + "═"*60)
    print("PARTIE 2 — Règles SWRL sur le KB des scientifiques")
    print("═"*60)

    from rdflib import Graph, Namespace
    EX = Namespace("http://example.org/")

    g = Graph()
    g.parse(kb_path, format="nt")
    print(f"\nKB chargé : {len(g)} triplets")

    # Règle 1 : transitiveInfluence
    print("\nRègle 1 (transitiveInfluence) :")
    print("  influencedBy(?p,?q) ∧ influencedBy(?q,?r) → indirectlyInfluencedBy(?p,?r)\n")

    new_triples_r1 = []
    for p, _, q in g.triples((None, EX.influencedBy, None)):
        for _, _, r in g.triples((q, EX.influencedBy, None)):
            if p != r:
                triple = (p, EX.indirectlyInfluencedBy, r)
                if triple not in new_triples_r1:
                    new_triples_r1.append(triple)
                    p_name = str(p).split("/")[-1]
                    r_name = str(r).split("/")[-1]
                    print(f"  → indirectlyInfluencedBy({p_name}, {r_name})")

    for triple in new_triples_r1:
        g.add(triple)

    if not new_triples_r1:
        print("  (aucun résultat)")

    # Règle 2 : sharedInstitution
    print("\nRègle 2 (sharedInstitution) :")
    print("  studiedAt(?p,?i) ∧ studiedAt(?q,?i) ∧ p≠q → sharedInstitution(?p,?q)\n")

    institution_map = {}
    for p, _, inst in g.triples((None, EX.studiedAt, None)):
        institution_map.setdefault(inst, []).append(p)

    new_triples_r2 = []
    for inst, scientists in institution_map.items():
        if len(scientists) >= 2:
            for i in range(len(scientists)):
                for j in range(i + 1, len(scientists)):
                    p, q = scientists[i], scientists[j]
                    triple = (p, EX.sharedInstitution, q)
                    if triple not in new_triples_r2:
                        new_triples_r2.append(triple)
                        p_name = str(p).split("/")[-1]
                        q_name = str(q).split("/")[-1]
                        inst_name = str(inst).split("/")[-1]
                        print(f"  → sharedInstitution({p_name}, {q_name}) [via {inst_name}]")

    for triple in new_triples_r2:
        g.add(triple)

    if not new_triples_r2:
        print("  (aucun résultat)")

    output_path = "kg_artifacts/expanded_reasoned.nt"
    g.serialize(output_path, format="nt")

    initial = len(g) - len(new_triples_r1) - len(new_triples_r2)
    print(f"\n{'─'*50}")
    print(f"Résumé raisonnement scientifiques :")
    print(f"  Triplets initiaux              : {initial}")
    print(f"  Triplets inférés (règle 1)     : {len(new_triples_r1)}")
    print(f"  Triplets inférés (règle 2)     : {len(new_triples_r2)}")
    print(f"  Triplets totaux après inférence: {len(g)}")
    print(f"  Fichier sauvegardé             : {output_path}")


if __name__ == "__main__":
    run_family_rules()
    run_scientist_rules()
    print("\nRaisonnement SWRL terminé.")
