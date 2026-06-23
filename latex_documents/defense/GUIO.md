# Guió de la defensa — OMR per a partitures de jazz

**Durada objectiu:** ~18–19 min (les 4 diapos de *Suport* són només per a preguntes).
**To:** proper però tècnic. Primera persona del singular per a la motivació personal (diapo 2); plural / impersonal («el sistema», «hem») per a la part tècnica.
**Consell:** no llegeixis les diapos — el tribunal ja les veu. Explica-les.

> Totes les xifres de la defensa coincideixen amb la memòria. Pots dir qualsevol número amb seguretat.

---

## 1 · Portada · (~20 s)

> «Bon dia. Sóc en Pol Casanovas i us presento el meu Treball de Fi de Grau: un sistema de **reconeixement òptic de música d'extrem a extrem per a partitures de jazz**, dirigit pel professor Manel Frigola. Us explico d'on surt, com funciona i què hem aconseguit.»

**Punt clau:** arrencar amb calma i mirar el tribunal.

---

## 2 · Per què aquest tema? · (~1 min 30 s) — *la teva història*

> «Fa molts anys que toco en **bandes de jazz i big bands**. I hi ha una cosa que passa sempre: la música arriba **en paper**. Quan hi ha vents, sovint et donen la part en **Si♭** o **Mi♭**, i te l'has de **transposar mentalment o a mà** a l'instant per encaixar amb la resta de la secció — o a l'inrevés. És lent, és fàcil equivocar-se, i ho repeteixes cada vegada.»
>
> «Això em va fer pensar: si aquella partitura fos **dades** en comptes d'una imatge, transposar-la seria immediat. I aquí apareix la pregunta del treball: **com pot un ordinador llegir una partitura?** Això és el reconeixement òptic de música, l'OMR — el que l'OCR va fer amb el text, però per a notació musical.»

**Punt clau:** la motivació és real i personal → enganxa el tribunal abans de la part tècnica.

---

## 3 · El problema: partitures només en paper o imatge · (~1 min 30 s)

> «El problema de fons és que molta música notada només existeix **en paper** o com a imatge escanejada. Els estàndards de jazz en són un cas clar: centenars de temes, a tots els faristols, però fins i tot els PDF són **només imatge** — no es poden cercar, ni editar, ni reproduir.»
>
> «Si ho poguéssim digitalitzar a notació simbòlica, desbloquejaríem la **transposició automàtica** per a instruments en Si♭, Mi♭ i Fa, reproducció MIDI, anàlisi harmònica de corpus i cerca. La figura de sota és una pàgina escanejada típica: el que veu, de fet, el sistema.»

**Punt clau:** el problema és concret i té valor pràctic. *(Si vols, pots posar* The Real Book *com a exemple en veu — la diapo ja no el nomena.)*

---

## 4 · Quatre reptes · (~1 min 30 s)

> «L'OMR és difícil per quatre motius. Primer, la **notació és bidimensional**: l'altura d'una nota depèn de la seva posició relativa a la clau, no del símbol en si. Segon, la **bretxa de domini**: un model entrenat amb imatges netes falla amb escanejos reals plens de soroll i distorsió. Tercer, l'**escassetat de dades**: no hi ha cap corpus etiquetat de *lead sheets*, així que les dades me les he hagut de generar. I quart, el **doble flux**: una *lead sheet* té melodia al pentagrama i acords a sobre; la majoria de sistemes només resolen el primer.»

**Punt clau:** aquests quatre reptes justifiquen totes les decisions que venen.

---

## 5 · Pipeline d'inferència · (~2 min)

> «El sistema és una cadena de cinc etapes. D'un PDF, primer **preprocessem** la pàgina, després **detectem els pentagrames**, i cada pentagrama passa per dos models en paral·lel: un **CRNN per a la melodia** i un altre **per als acords**. Finalment, un **corrector gramatical determinista** reconstrueix les barres de compàs i ho serialitzem tot a JSON.»
>
> «Un detall important és la **injecció de capçalera**: als reculls de jazz, tots els pentagrames menys el primer no tenen clau, armadura ni compàs. Com que el model s'ha entrenat sempre amb pentagrames complets, abans del CRNN de melodia hi afegim una plantilla amb la capçalera — així el model sempre veu el que espera. Tot plegat triga **114 mil·lisegons per pàgina** en una RTX 3060.»

**Punt clau:** modular, dos fluxos en paral·lel, injecció de capçalera = la idea pròpia.

---

## 6 · Decisions de disseny · (~2 min) — *analitza alternatives (rúbrica)*

> «Per a l'arquitectura vaig considerar tres alternatives. Un **pipeline modular** clàssic necessita segmentació símbol a símbol, i no tenia etiquetes per a això. Un **transformer** com l'SMT dóna molt bon SER, però amb un cost de còmput més de deu vegades superior — fora del meu pressupost d'una sola GPU. I un **seq2seq amb atenció** afegeix complexitat sense un benefici clar en lectura monofònica. Per això vaig triar **CRNN–CTC**, que aprèn l'alineament sol i és eficient en dades i en còmput.»
>
> «Per al format de sortida vaig descartar **MusicXML** — és un arbre jeràrquic, incompatible amb el flux de tokens del CTC — i vaig triar **LMX**: tokens plans i atòmics, un vocabulari d'uns 100 símbols i una gramàtica prou regular per aplicar-hi un corrector determinista.»

**Punt clau:** cada decisió és una tria justificada enfront d'alternatives rebutjades.

---

## 7 · Dades sintètiques i simulació d'escaneig · (~1 min 30 s)

> «Com que no hi havia dades, les vaig construir. Vaig agafar **PrIMuS** —gairebé 88.000 fragments— i el vaig re-renderitzar en estil **LilyJAZZ**, la tipografia de jazz. Després de filtrar pel meu domini queden **46.089 mostres**, partides en train, validació i test sense fuga.»
>
> «La clau per tancar la bretxa de domini és la **simulació d'escaneig**: a cada mostra li aplico soroll, distorsió de perspectiva, taca de tinta, esgrogueïment… La figura de la dreta mostra una mostra neta i tres variants escanejades. El criteri que em vaig posar era que la diferència entre net i escanejat fos menys d'un punt percentual — i ha quedat en **0,06**.»

**Punt clau:** dades sintètiques + augmentació = la resposta a l'escassetat i a la bretxa de domini.

---

## 8 · Resultats principals · (~2 min)

> «Aquests són els resultats sobre les 4.608 mostres de test. El **SER agregat** és de **1,23 %** sobre l'escaneig simulat i 1,17 % sobre net — per sota de l'objectiu del 3 % que m'havia fixat. Però el més rellevant és el **SER melòdic**: **0,14 %**. Vol dir que el model encerta el **99,86 %** dels tokens de to, durada i alteració. I gairebé tres de cada quatre pentagrames es transcriuen **perfectes**, sense cap error.»
>
> «A més, la diferència entre net i escanejat és de només **0,06 punts**, cosa que confirma que la simulació d'escaneig funciona.»

**Punt clau:** SER melòdic 0,14 % és la xifra estrella — el contingut musical és quasi perfecte.

---

## 9 · On són els errors? · (~1 min 30 s)

> «Si el SER agregat és 1,23 % però el melòdic és 0,14 %, on és la diferència? Aquí. Gairebé el **90 %** de les edicions són **barres de compàs i lligadures** — tokens estructurals. El to i l'octava són pràcticament absents dels errors, i les substitucions, que són les que afecten el contingut real, són només el **3,4 %**.»
>
> «La conclusió és que el model **llegeix bé la música**; de tant en tant col·loca malament una barra de compàs, i això ja ho recupera el corrector gramatical aigües avall.»

**Punt clau:** els errors són estructurals i recuperables, no melòdics.

---

## 10 · Comparació qualitativa: Satin Doll · (~1 min 30 s)

> «Per situar-ho davant d'una eina existent, vaig passar la mateixa pàgina real d'*Satin Doll* per **Audiveris**, el motor lliure de referència, i pel meu sistema. Audiveris detecta la clau correcta només a **2 de 9 pentagrames** —la resta van per defecte a clau de fa, amb les octaves desplaçades— i **no recupera cap acord**: la sortida és inservible.»
>
> «El meu sistema reconeix **7 dels 9 pentagrames**, tots en clau de sol, recupera **36 acords**, i rebutja correctament les dues regions que no són música: el títol i la línia de crèdits. La diferència ve de les dues decisions que Audiveris no pren: la injecció de capçalera i el flux dedicat d'acords.»

**Punt clau:** comparació honesta i qualitativa — no hi ha *ground truth*, però el contrast és clar.

---

## 11 · Objectius assolits · (~1 min)

> «Repassant els objectius: l'objectiu general i sis dels set específics estan **assolits** — l'estat de l'art, el dataset, la simulació d'escaneig, el model i la seva avaluació, l'anàlisi d'errors i el *pipeline* complet amb API. El **setè és parcial**: he fet el *fine-tuning* amb 36 pentagrames reals i millora la transcripció qualitativament, però amb només 36 no en podia reservar prou per a un test real estable, així que la xifra quantitativa queda pendent.»

**Punt clau:** sigues honest amb l'O7 — la sinceritat sobre l'abast puntua més que sobrevendre.

---

## 12 · Conclusions i treball futur · (~1 min 30 s)

> «En resum: he construït un sistema OMR d'extrem a extrem per a jazz, amb un SER agregat de l'1,23 %, un SER melòdic del 0,14 %, gairebé un 73 % de transcripcions perfectes, i un flux d'acords que baixa al 5,9 % de CER després del *fine-tuning*.»
>
> «Els tres passos següents són clars: **etiquetar més de 200 pentagrames reals** per tancar quantitativament l'objectiu 7; afegir **tresets** al vocabulari, que és el buit més gran; i un **detector de pentagrames après** per als escanejos més difícils. I tornant al principi: tot això obre la porta a digitalitzar *The Real Book* i transposar-lo automàticament per a qualsevol instrument.»

**Punt clau:** tanca tornant a la motivació inicial — narrativa circular.

---

## 13 · Competències tècniques · (~45 s)

> «Finalment, el treball cobreix les dues competències de Computació que vaig declarar, totes dues en profunditat: **CCO1.1**, en l'anàlisi d'estratègies algorísmiques i rendiment —les decisions d'arquitectura i descodificació—, i **CCO2.4**, en aprenentatge automàtic sobre grans volums de dades.»

**Punt clau:** menció explícita — el tribunal busca aquest mapatge.
*(La diapo encara cita «fuga de dades detectada i corregida» com a evidència de la CCO2.4; si vols, ho trec — veure nota al final.)*

---

## 14 · Referències · (~10 s)

> (Passa-la de pressa.) «Aquestes són les referències principals del treball.»

---

## 15 · Gràcies · (~20 s)

> «I això és tot. **Gràcies** per la vostra atenció — quedo a la vostra disposició per a les preguntes.»

**Punt clau:** acaba amb seguretat i en silenci; deixa la diapo de xifres a la vista per a les preguntes.

---

## Diapos de suport (per a preguntes) — on saltar

| Si pregunten sobre… | Vés a |
|---|---|
| Corba / convergència de l'entrenament | **Suport · Corba d'entrenament** |
| Reconeixement d'acords en detall (CER, zero-shot vs *fine-tune*) | **Suport · Reconeixement d'acords** |
| Velocitat / cost / desglossament per etapa | **Suport · Latència** |
| «Quin és el pitjor cas?» / errors melòdics | **Suport · Un cas melòdic difícil** |

### Preguntes probables i resposta curta
- **Per què CRNN i no transformer?** → Pressupost de còmput (1 RTX 3060) i eficiència de dades; el transformer dóna millor SER però a >10× cost. (diapo 6)
- **Per què el test real és tan petit?** → Només 36 pentagrames reals etiquetats; tots calen per entrenar, no en queden prou per a un test estable. És la limitació principal i el primer treball futur. (O7)
- **Què és el SER?** → *Symbol Error Rate*: distància d'edició entre la seqüència predita i la real, dividida per la longitud. L'equivalent del WER de la parla.
- **Com garanteixes que no hi ha fuga entre train i test?** → La partició es fa per **id de mostra** i totes les variants (neta + escanejades) d'una mateixa peça queden al **mateix costat** del split; les variants derivades són **només a train**. *(De fet, a mig projecte vaig detectar una fuga per aquest motiu, la vaig corregir redissenyant el protocol de partició, i totes les xifres reportades són posteriors a la correcció.)*
- **El pitjor cas?** → El pitjor cas absolut (SER **0,112**, mostra `210017589-1_51_1`) és un **desplaçament de barra**, no un error melòdic. La diapo de suport mostra el cas melòdic més il·lustratiu (rang 3, SER **0,089**): **confusió d'octava** en notes greus dins patrons de semicorxera en Mi♭ major. Els errors melòdics visibles són poc freqüents.
- **Per què LMX i no MusicXML?** → MusicXML és un arbre jeràrquic, incompatible amb el flux atòmic del CTC; LMX són tokens plans. (diapo 6)

---

> **Nota** — La diapositiva dedicada a la fuga de dades s'ha tret del recorregut principal. Segueix sent a la memòria i la pots explicar de viva veu si et pregunten per la higiene del split (resposta preparada a dalt). La diapo 13 (Competències) encara la menciona en una línia com a evidència de la CCO2.4; si la vols treure també, digues-m'ho.
