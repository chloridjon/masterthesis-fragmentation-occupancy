> Technical calculation record from the thesis. Generated-table links refer to local outputs. Use the public workflow for current execution; earlier pilot replication does not replace the final 8/16 design.

# Self Colonisation: Prüfung im separat ausgewählten Antwortbereich

2. November 2026. Auswertung für TODO 5-05; keine eingefügte Thesispassage.

## Auftrag und Umfang

Der Autor verlangt einen für q=1 analog bestimmten Parameterbereich, bevor
beurteilt wird, ob dort negative Szenariounterschiede auftreten. Ein gröberes
Gitter und weniger Wiederholungen als für die Hauptanalyse genügen. Die
65 unveränderten q=0-Profile allein beantworten diese Frage nicht.

Bereits berechnete, numerisch geprüfte q=1-Daten erlauben diese erste
Auswertung ohne neue Modellläufe: sieben Szenarien, acht vollständige
gematchte Landschaftsläufe, fünf Waldanteile (0.025, 0.2, 0.4, 0.6, 0.825).
Die Prüfung nutzt zuerst fünf, dann alle acht vorhandenen Läufe und lässt
zusätzlich jeden vollständigen Lauf einzeln weg. Durchgehend gelten die
ursprünglichen Ausschlüsse für Landschaften mit weniger als zwei Patches
und mindestens drei gültige Läufe pro Szenario-H-Zelle.

## Analoges Auswahlverfahren

Die separat für q=1 ausgewählte Arbeitsrechteckgrenze wurde unverändert aus
dem geprüften Entscheidungsprotokoll übernommen: Halbdistanz 1–512 Zellen,
r von 10^-1.5 (0.0316228) bis 10^6.75 (5,623,413.25). Sie beruht auf der
gleichen Übergangshüllen-/Dispersalplateau-Regel wie die q=0-Rechteckauswahl.
Der q=1-Rand wurde bereits über die übrigen gespeicherten H validiert.
Die fünf H bleiben eine Einschränkung der Plateauentscheidung.

Innerhalb dieses Rechtecks liegen 340 vorhandene Gitterpunkte: zehn
Dispersalhalbdistanzen in Verdopplungsschritten und 34 r-Werte in Schritten
von 0.25 auf der log10-Skala. Auf diese Punkte wird zusätzlich die gleiche
Antwortbereichsregel wie im endgültigen q=0-Design angewandt:

- L: kleinstes Landschafts-5%-Quantil über geeignete Szenario-H-Zellen.
- U: größtes Landschafts-95%-Quantil über geeignete Szenario-H-Zellen.
- Beibehalten, falls L < 0.95 und U > 0.05; lineare Quantile wie im Hauptdesign.

Die Auswahl verwendet keine Szenariokontraste. Sie schließt globale
Sättigung aus; sie verlangt weder einen vollständigen Übergang bei jedem
Waldanteil noch in jedem Szenario. Es findet keine Paarung von q=0- und
q=1-Profilen nach relativer Position oder Belegung statt.

**Geltungsgrenze:** Die vorhandenen fünf H liefern Zeugen für beide
Ungleichungen. Ein so beibehaltener Punkt erfüllt damit die globale
Antwortbereichsregel auch dann, wenn weitere H ergänzt werden (bei derselben
Laufstichprobe). Nicht beibehaltene Punkte sind für die ursprüngliche
Alle-H-Regel weiterhin ungeklärt: Sie dürfen nicht allein auf dieser Basis
als über alle 88 H gesättigt gelten. Wir bestimmen deshalb eine konservative
diskrete innere Teilmenge, keine vollständig rekonstruierte Bereichsgrenze.
Zwischen Gitterpunkten wird keine glatte Grenze interpoliert.

Mit fünf Läufen werden 225 Punkte beibehalten, mit acht Läufen 235.
223 bzw. 229 Punkte bleiben in jeder vollständigen Laufweglassung erhalten.
Die Anzahlen sind Gitterdiagnostik, keine Schätzung des Anteils ökologischer
Arten oder der kontinuierlichen Parameterfläche.

## Befund für access based, rho=0.999

Kontrast: Szenariomedian minus only-random-Median, innerhalb von q=1.
Die fünf Prozentpunkte sind die bereits verwendete deskriptive
Vergleichsschwelle, kein Signifikanz- oder biologischer Grenzwert.

| H | Stärkster negativer Kontrast, 8 Läufe | Punkte unter -0.05 / 235 |
|---|---:|---:|
| 0.025 | numerisch null | 0 |
| 0.2 | kein negativer Kontrast | 0 |
| 0.4 | -0.00277 | 0 |
| 0.6 | -0.01953 | 0 |
| 0.825 | -0.07871 | 6 |

Die sechs deutlich negativen Punkte bei H=0.825 liegen bei Halbdistanz
1 Zelle und r=1,778.28 bis 31,622.78. Das Minimum liegt bei r=10,000:
only random hat Median 0.572700, access based 0.493991. Dieser Punkt erfüllt
die Auswahlregel (L=0, U=0.581713); er ist keine nachträgliche Ausnahme.

Mit nur fünf Läufen beträgt das Minimum bei H=0.825 -0.09319;
acht Gitterpunkte liegen unter -0.05. Bei jeder Weglassung eines der acht
Läufe bleiben vier bis sieben solche Punkte bestehen; das jeweilige
Minimum liegt zwischen -0.09292 und -0.06450. Bei H=0.6 liegt dagegen
auch mit fünf Läufen und bei jeder Laufweglassung kein Kontrast unter -0.05.
Über alle Szenarien/H beträgt die größte Kontraständerung durch Weglassen
eines Laufs 0.04811 bei acht bzw. 0.08461 bei fünf Läufen. Diese Diagnostik
belegt keine allgemeine Stichprobenpräzision.

Andere Szenarien behalten ebenfalls negative Bereiche. Insbesondere ist
bei H=0.6 für edge random/rho=0.999 ein Minimum von -0.11738 vorhanden;
die Aussage für access based darf nicht auf alle Szenarien ausgedehnt werden.

## Konsequenz für TODO 5-05

Die feste-Parameter-Vorschau bleibt eine Mechanismusprüfung. Die neue
Bereichsprüfung stützt für access based/rho=0.999 bei H=0.6 das Ausbleiben
der zuvor deutlich negativen Kontraste auf dem geprüften Gitter. Sie stützt
**nicht** die pauschale Behauptung, Self Colonisation beseitige alle negativen
Szenariounterschiede. Bei hohem Waldanteil bleiben selbst im separat
ausgewählten q=1-Antwortbereich solche Unterschiede erhalten.

Fünf H, ein endliches Gitter und acht Läufe begrenzen die Aussage.
Insbesondere werden keine Abwesenheit schmaler Regionen zwischen diesen
Punkten und keine Konfidenzintervalle behauptet. Eine neue q=1-Verteilung
mit 256 Profilen, ein Vogel-Neufit oder eine Umrechnung r0 -> r1 sind
für diese Prüfung weder durchgeführt noch erforderlich.

TODO 5-05 bleibt offen, bis Interpretation und gegebenenfalls neue
Abbildung/Caption mit dem Autor abgestimmt sind. Kein Manuskripttext
eingefügt, keine neue Draftnummer; Arbeitsfassung bleibt v3.

## Prüfungen und Reproduktion

- 280 Original-CSV-Dateien und ihre Receipts per SHA256 geprüft;
  unabhängige numerische Ausgangsaudits gegen ihre gespeicherten Hashes geprüft.
- 95,200 Antwortzeilen im Rechteck auf Identitäten, Duplikate,
  Patchausschlüsse, Belegungsgrenzen und gespeicherte Gleichgewichtsresiduen geprüft.
- 178,500 Zusammenfassungszeilen einschließlich aller 13 vollständigen
  Laufweglassungen unabhängig aus Originalzeilen mit NumPy reproduziert.
- 16 ursprüngliche Patchvektoren für das stärkste fokale Beispiel per
  Hash, Patchanzahl, Wertebereich und Mittelwert überprüft; Kontrast aus
  den Vektormitteln reproduziert.
- Keine neuen Modellläufe, Landschaften, Bootstrap-Intervalle oder Fits.

Aus dem Repository-Hauptverzeichnis mit vorhandener .venv ausführen:

```sh
.venv/bin/python Masterarbeit_final/code/analyses/02_metapop/02_04_robustness/13_self_colonization_response_domain.py
.venv/bin/python Masterarbeit_final/code/analyses/02_metapop/02_04_robustness/13_self_colonization_response_domain_review.py
```

Konfiguration: gleichnamiger Analyseordner unter `config`; Tabellen und
maschinenlesbare Prüfprotokolle unter `tables` und `logs` dieses Ordners.
