# knowledge_eng

Una giovane banca sta crescendo rapidamente acquisendo sempre più nuovi clienti. La maggior parte di questi sono creditori, ovvero presentano depositi di varie dimensioni presso la banca. Il numero di clienti che sono anche debitori risulta relativamente piccolo e la banca è interessata ad espandere rapidamente tale insieme in modo da ingrandire il business dei prestiti e quindi aumentare i guadagni grazie agli interessi. Una campagna lanciata l’anno precedente per convertire clienti creditori in clienti aventi anche un prestito (e quindi debitori) ha portato ad un grado di conversione di oltre il 9%. Questo ha incoraggiato il dipartimento di marketing alla creazione di una nuova campagna mirata ai clienti creditori che più probabilmente possano avviare un prestito con la banca. L’obiettivo del dipartimento è quindi quello di creare un modello che li aiuti a identificare potenziali clienti per l’acquisto di prestiti, in modo da aumentare i guadagni globali e diminuire le spese per la prossima campagna.

  1.2 Contenuto del Dataset
Il dataset utilizzato riporta informazioni relative alla scorsa campagna per 5000 clienti. Di seguito sono riportate le feature associate a ciascun cliente:
  1.	ID (numeri interi): numero identificativo
  2.	Age (numeri interi): anni compiuti
  3.	Experience (numeri interi): anni di esperienza professionale
  4.	Income (numeri interi): entrate annue in k (8 = 8000)
  5.	ZIP Code (numeri interi): codice postale residenza cliente
  6.	Family (numeri interi): numero membri in famiglia
  7.	CCAvg (numeri decimali): spesa media mensile da carta di credito
  8.	Education (numeri interi da 1 a 3): livello di educazione (1: non laureato, 2: laureato, 3: avanzato/professionale)
  9.	Mortgage (numeri interi): valore del mutuo sulla casa in k (se presente)
  10.	Securities Account (booleano): indica se il cliente abbia un conto titoli con la banca o meno
  11.	CD Account (boolean): indica se il cliente abbia un certificato di deposito con la banca o meno
  12.	Online (boolean): indica se l’utente utilizzi il servizio di internet banking o meno
  13.	CreditCard (boolen): indica se il cliente usi una carta di credito rilasciata dalla banca o meno
  14.	Personal Loan (boolean): indica se il cliente abbia accettato il prestito offerto dalla scorsa campagna e rappresenta il campo target su cui avverrà il traning, il test e la       valutazione dei vari modelli che affronteremo.
