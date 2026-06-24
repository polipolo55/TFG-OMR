# Guió de la defensa — OMR per a partitures de jazz

**Durada objectiu:** ~18–19 min (+ ~1–2 min si reprodueixes la demostració). Les diapos de *Suport* són només per a preguntes.
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

## 10·5 · Demostració del sistema · (~1–2 min) — *opcional, segons temps*

> «I perquè no quedi tot en xifres, una demostració ràpida del sistema funcionant d'extrem a extrem: d'una pàgina escanejada del *Real Book* fins a la transcripció simbòlica i el JSON.»

*(Clica el botó **▶ Veure la demostració** — obre el vídeo al navegador. Mantén el clip curt, ~1 min.)*

**Punt clau:** el vídeo fa tangible tot el *pipeline*. Si vas just de temps, mostra la diapo i ofereix reproduir-lo a les preguntes — el QR permet que el tribunal el vegi pel seu compte.

> **Vídeo (no llistat):** https://youtu.be/JAkTSFvwVQ4

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

> **Avaluació (Art. 21):** el tribunal de la fita final puntua les **competències tècniques (60 %)**. Les **transversals (40 %)** les avaluen el GEP i el director al llarg de les tres fites — no es re-puntuen a la defensa. Així doncs, les diapos de suport transversals (gestió, pressupost, sostenibilitat, legal) són per **respondre amb solidesa si pregunten** (i pel director), no per al recorregut puntuat.

| Si pregunten sobre… | Vés a |
|---|---|
| Corba / convergència de l'entrenament | **Suport · Corba d'entrenament** |
| Reconeixement d'acords en detall (CER, zero-shot vs *fine-tune*) | **Suport · Reconeixement d'acords** |
| Velocitat / cost / desglossament per etapa | **Suport · Latència d'inferència** |
| «Quin és el pitjor cas?» / errors melòdics | **Suport · Un cas melòdic difícil** |
| Arquitectura / capes / CRNN–CTC en detall | **Suport · Arquitectura del model** |
| LMX / tokens / codificació de sortida | **Suport · LMX: gramàtica i exemple** |
| Fuga de dades / *leakage* / com es fa el *split* | **Suport · Higiene de dades** |
| SER brut vs net / paper del corrector gramatical | **Suport · SER brut vs. contingut** |
| Planificació / metodologia / desviacions / risc | **Suport · Gestió i metodologia** |
| Cost / pressupost / hores | **Suport · Pressupost** |
| Sostenibilitat / energia / CO₂ / ètica | **Suport · Sostenibilitat** |
| Drets d'autor / Real Book / llicències / RGPD | **Suport · Aspectes legals** |

### Preguntes probables i resposta curta

**Disseny i tècnica**
- **Per què CRNN i no transformer?** → Pressupost de còmput (1 RTX 3060) i eficiència de dades; el transformer dóna millor SER però a >10× cost. (diapo 6)
- **Has entrenat realment un transformer amb les teves dades?** → No: hauria consumit tot el pressupost d'una sola GPU al voltant del qual s'escala el projecte — i això és precisament l'argument. La justificació és per adjudicació d'enginyeria (autoregressiu = *exposure bias* + descodificació seqüencial + més dades; polifonia fora d'abast), no per *benchmark*. Que el *beam* només guanyi <0,01 pp mostra que la maquinària autoregressiva és innecessària aquí. Una comparació amb transformer entrenat és el contrast net però inassequible. (diapo 6)
- **On és la contribució algorísmica profunda (CCO1.1)?** → En les decisions adjudicades: ResNet18 de passes **asimètriques** que preserva l'eix temporal del CTC a ÷4 exacte; *beam* vs voraç quantificat (8× cost, <0,01 pp); corrector gramatical determinista vs descodificador restringit/FST. La contribució és el *pipeline* específic de domini (LilyJAZZ + simulació calibrada + corrector + doble flux), no un CRNN nou. (Suport · Arquitectura)
- **Per què LMX i no MusicXML?** → MusicXML és un arbre jeràrquic, incompatible amb el flux atòmic del CTC; LMX són tokens plans. (diapo 6, Suport · LMX)
- **Què és el SER?** → *Symbol Error Rate*: distància d'edició entre la seqüència predita i la real, dividida per la longitud. L'equivalent del WER de la parla.

**Resultats i validesa**
- **Per què el test real és tan petit?** → Només 36 pentagrames reals etiquetats; tots calen per entrenar, no en queden prou per a un test estable. És la limitació principal i el primer treball futur. (O7)
- **Reportes SER pre- o post-corrector? No amagues errors?** → El SER agregat és la sortida **crua** del CRNN, **abans** del corrector i sobre pentagrames aïllats — els errors estructurals **compten en contra**. El SER melòdic (0,14 %) és una mètrica separada (només contingut). Es mantenen les dues per jutjar-les per separat. (Suport · SER brut vs. contingut)
- **Una sola llavor, cap variància?** → Cada run són ~13 h, multi-llavor fora de pressupost (i ho dic). *Proxies*: el *plateau* de validació és estret (±0,02 pp) i l'avaluador independent coincideix amb l'estimació en bucle. No sobreinterpreto els 0,06 pp: és una cota d'una banda, no un efecte precís. Multi-llavor = treball futur.
- **Satin Doll no té *ground truth* — no és *cherry-picking*?** → És qualitatiu i mai reporto una taxa d'error head-to-head. No és triat: és una pàgina real del domini exacte i el contrast és **estructural** (injecció de capçalera + flux d'acords), no numèric. Comparació quantitativa multi-pàgina i multi-motor = treball futur.
- **El SER melòdic 0,14 % i les confusions de to?** → Les substitucions (únic tipus que canvia el contingut) són el 3,4 % de 3.506 edicions ≈ ~119 tokens en tot el test; el parell líder (B→G) surt 8 cops i la resta cau ràpid. Un músic notaria un salt d'octava en un passatge greu (és el pitjor cas de suport), però són rars i localitzats.
- **La porta de rebuig per CTC és fràgil (re-ajust manual)?** → Limitació documentada: 4 llindars geomètrics independents del *checkpoint* + 1 llindar CTC dependent del *checkpoint*, ajustat a mà i **preservant música** (verificat en pàgines reals). A Satin Doll rebutja el títol i els crèdits i manté els 7 pentagrames. El detector OOD après és treball futur.

**Transversals (gestió · legal · sostenibilitat)**
- **Com garanteixes que no hi ha fuga entre train i test?** → Partició per **id de mostra**; totes les variants (neta + escanejades) d'una peça al **mateix costat**; diversitat derivada **només a train**. *(A mig projecte vaig detectar una fuga d'aquest tipus, la vaig corregir redissenyant el protocol, i totes les xifres són posteriors.)* (Suport · Higiene de dades)
- **The Real Book té drets d'autor (Hal Leonard) — és legal?** → Sí. El CRNN de melodia s'entrena **només amb dades sintètiques**; els escanejos reals només per avaluació i un petit *fine-tuning* d'acords, **mai redistribuïts**; els pesos guarden **tokens, no píxels**. Base: RD Leg. 1/1996 art. 32 (docència/recerca) + Directiva UE 2019/790 art. 3 (mineria de text i dades). El llançament exclou tot material derivat del Real Book. (Suport · Aspectes legals)
- **PrIMuS és per a recerca no comercial — i un ús comercial?** → S'usa sense modificar i només per a entrenament de recerca dins el TFG; els artefactes alliberats són *checkpoints* de recerca amb llicència oberta, no un producte. «Digitalitzar el Real Book» és la motivació i una direcció de recerca, no un servei comercial; un desplegament comercial requeriria re-llicenciar les dades i queda fora d'abast. LilyJAZZ ve amb LilyPond (només per renderitzar). (Suport · Aspectes legals)
- **D'on surten els 62,5 kWh / 14,4 kg CO₂e?** → Estimació de projecte (potència nominal × temps, factor REE/MITECO ~0,23 kg/kWh), descrita com a «grossera però reproduïble». El run final de 90 èpoques ≈ 13 h; mitigacions: ResNet18, precisió mixta, *early stopping*, reús de maquinari. Manufactura i fi de vida queden **fora** (qualitatius). (Suport · Sostenibilitat)
- **Quant ha costat?** → 16.097,16 € pressupostats: ~85 % temps d'enginyeria (13.615,57 €), 250,69 € generals, 15 % contingència, imprevistos de risc; 509 h vs 540 h de l'Art. 17; programari **0 €** (codi obert); RTX 3060 amortitzada 125,69 €. (Suport · Pressupost)
- **Quina metodologia i quines desviacions?** → Iterativa-incremental, sprints de 4 setmanes per WP, bucle dades→entrenament→anàlisi. Dues replanificacions: sprints GEP 2→4 set., i el flux d'acords absorbit sense ampliar pressupost. Registre de 6 riscos + control per valor guanyat. (Suport · Gestió i metodologia)
- **Si tornessis a començar, què faries diferent?** → Construir el flux d'acords **en paral·lel** amb la melodia des del principi (no tard), per donar més iteracions al *fine-tuning* real; i avançar l'etiquetatge real perquè existís un test real (tancar O7). Vaig aprendre que la bretxa sintètic→real és **desigual** (la notació transfereix molt millor que el text d'acords) i que la higiene del *split* pot inflar-ho tot fins que l'audites per construcció.

> **Consell de presentació (diapo 14, Competències):** en parlar-ne, anomena explícitament la **integració de coneixements** del grau — aprenentatge profund, algorísmia, arquitectura de programari i disseny d'API — perquè el tribunal marqui aquest criteri de la rúbrica (Art. 10.4), no només els dos codis CCO.

---

> **Nota — diapo 14 (Competències):** mantén la línia «fuga de dades detectada i corregida». És la millor evidència d'iniciativa i autocorrecció (criteri transversal) i ara està reforçada per la diapo de suport **Higiene de dades**. L'auditoria contra la normativa recomana **no treure-la**.
