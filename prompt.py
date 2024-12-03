prompts_fewshot_template_cot=f'''
### Instruction:
Predict the relation between Arg1 and Arg2, just choose one or two label from :[Temporal, Comparison, Contingency, Expansion], no need to explain

### Input:
<ARG1>Was this why some of the audience departed before or during the second half?
Or was it because Ms. Collins had gone</ARG1><ARG2>Either way it was a pity</ARG2>

### Response:
Let's think step by step, and give a summary at the end:

Identifying the Arguments:

ARG1 questions the reason for the audience leaving during a performance.
ARG2 comments on the situation, noting it as unfortunate regardless of the reason.
Determining the Relationship:

ARG2 extends the scenario presented in ARG1 by providing an evaluative remark on the outcome.
Reason for Expansion:

Without specifying or resolving the reasons posed in ARG1, ARG2 broadens the discussion by adding a perspective on the overall situation.
Summary: The classification as "Expansion" is because ARG2 broadens the discussion initiated in ARG1 with an evaluative statement, adding to the context without altering its course.

Expansion

### Instruction:
Predict the relation between Arg1 and Arg2, just choose one or two label from :[Temporal, Comparison, Contingency, Expansion], no need to explain

### Input:
<ARG1>The rest of the concert was more straight jazz and mellow sounds written by Charlie Parker, Ornette Coleman, Bill Douglas and Eddie Gomez, with pictures for the Douglas pieces</ARG1><ARG2>It was enjoyable to hear accomplished jazz without having to sit in a smoke-filled club, but like the first half, much of it was easy to take and ultimately forgettable</ARG2>

### Response:
Let's think step by step, and give a summary at the end:

Identifying the Arguments:

ARG1 describes the music played in the second half of a concert.
ARG2 expresses enjoyment of the music but notes it was ultimately forgettable.
Determining the Relationship:

ARG2 extends ARG1 by evaluating the concert experience, without contradicting or providing a causal link.
Reason for Expansion:

ARG2 adds to the description in ARG1 by sharing a personal reaction, fitting the "Expansion" category.
Summary: The classification as "Expansion" is due to ARG2 adding an evaluative perspective to the concert's description in ARG1, enhancing the information without altering its course or presenting an opposing viewpoint.

Expansion

### Instruction:
Predict the relation between Arg1 and Arg2, just choose one or two label from :[Temporal, Comparison, Contingency, Expansion], no need to explain

### Input:
<ARG1>Mr. Mayor's hope that references to "press freedom" would survive unamended seems doomed to failure</ARG1><ARG2>the current phrasing is "educating the public and media to avoid manipulation</ARG2>

### Response:
Let's think step by step, and give a summary at the end:
Identifying the Arguments:

ARG1 expresses concern about changes to "press freedom" references.
ARG2 presents the current phrasing regarding media education.
Determining the Relationship:

ARG2 specifies the nature of the amendments feared in ARG1, showing a cause-and-effect link.
Reason for Contingency:

The relationship is "Contingency" because ARG2 illustrates the specific outcome or effect (the amendment details) of the situation mentioned in ARG1 (concern about preserving "press freedom").
Summary: The classification as "Contingency" comes from ARG2 demonstrating the specific change or consequence that ARG1 was concerned about, indicating a cause-and-effect relationship between the two arguments.

Contingency

### Instruction:
Predict the relation between Arg1 and Arg2, just choose one or two label from :[Temporal, Comparison, Contingency, Expansion], no need to explain

### Input:
<ARG1>The Orwellian "New World Information Order" would give government officials rights against the press</ARG1><ARG2>journalists would be obliged to kowtow to their government, which would have licensing and censorship powers and, indeed, duties to block printing of "wrong" ideas</ARG2>

### Response:
Let's think step by step, and give a summary at the end:

Identifying the Arguments:

ARG1 mentions a proposed "New World Information Order" affecting press freedom.
ARG2 describes consequences for journalists under this order.
Determining the Relationship:

ARG2 provides specific effects of the "New World Information Order" mentioned in ARG1.
Reason for Contingency:

ARG2 shows the direct outcomes of the situation in ARG1, indicating a cause-and-effect link.
Summary: The classification as "Contingency" is due to ARG2 illustrating the direct consequences for journalists of the "New World Information Order" introduced in ARG1.

Contingency

### Instruction:
Predict the relation between Arg1 and Arg2, just choose one or two label from :[Temporal, Comparison, Contingency, Expansion], no need to explain

### Input:
<ARG1>He will be in charge of research, equity sales and trading, and the syndicate operation of Rothschild</ARG1><ARG2>Mr. Conlon was executive vice president and director of the equity division of the international division of Nikko Securities Co</ARG2>

### Response:
Let's think step by step, and give a summary at the end:

Identifying the Arguments:

ARG1 states a new responsibility for an individual at Rothschild.
ARG2 provides background on Mr. Conlon's previous position at Nikko Securities Co.
Determining the Relationship:

ARG2 describes Mr. Conlon's past role, which is logically prior to the new role mentioned in ARG1.
Reason for Temporal:

The sequence implies a before-and-after relationship, with ARG2 setting a temporal context for ARG1.
Summary: The classification as "Temporal" is due to ARG2 providing a background context that precedes the new appointment described in ARG1, indicating a time-based sequence of events.

Temporal

### Instruction:
Predict the relation between Arg1 and Arg2, just choose one or two label from :[Temporal, Comparison, Contingency, Expansion], no need to explain

### Input:
<ARG1>He gave up seven hits, walked five and didn't get a decision</ARG1><ARG2>Arm troubles forced him back to the minors the next year</ARG2>

### Response:
Let's think step by step, and give a summary at the end:
Identifying the Arguments:

ARG1 describes a baseball player's performance in a game.
ARG2 explains the player's subsequent demotion to the minors due to arm troubles.
Determining the Relationship:

ARG2 follows ARG1 in time, linking a poor performance to later consequences for the player.
Reason for Temporal:

The events in ARG2 (demotion due to arm troubles) occur after the events in ARG1 (game performance), suggesting a sequence.
Summary: The "Temporal" classification reflects ARG2 happening after ARG1, with the player's performance leading to future events affecting his career.

Temporal

### Instruction:
Predict the relation between Arg1 and Arg2, just choose one or two label from :[Temporal, Comparison, Contingency, Expansion], no need to explain

### Input:
<ARG1>Seats currently are quoted at $400,000 bid, $425,000 asked</ARG1><ARG2>The record price for a full membership on the exchange is $550,000, set March 9</ARG2>

### Response:
Let's think step by step, and give a summary at the end:
Identifying the Arguments:

ARG1 mentions the current bid and ask prices for seats on an exchange.
ARG2 states the record price for full membership on the same exchange.
Determining the Relationship:

ARG2 compares the current prices mentioned in ARG1 to a historical high.
Reason for Comparison:

The inclusion of a record price in ARG2 serves to compare or contrast with the current prices provided in ARG1.
Summary: The classification as "Comparison" is due to ARG2 presenting a historical price point that serves as a benchmark for the current prices mentioned in ARG1, facilitating a direct comparison.

Comparison

### Instruction:
Predict the relation between Arg1 and Arg2, just choose one or two label from :[Temporal, Comparison, Contingency, Expansion], no need to explain

### Input:
<ARG1>In late trading, the shares were up a whopping 122 pence ($1.93) -- a 16.3% gain -- to a record 869 pence on very heavy volume of 9.7 million shares</ARG1><ARG2>In the U.S. over-the-counter market, Jaguar shares trading as American Depositary Receipts closed at $13.625, up $1.75</ARG2>

### Response:
Let's think step by step, and give a summary at the end:
Identifying the Arguments:

ARG1 details a significant increase in share price on one market.
ARG2 describes the performance of the same shares in a different market.
Determining the Relationship:

Both ARG1 and ARG2 report on the financial performance of the same shares, but in different trading venues.
Reason for Comparison:

The juxtaposition of share price movements in two different markets naturally invites a comparison between the two scenarios.
Summary: The "Comparison" classification stems from ARG1 and ARG2 providing information on the performance of the same shares in two distinct markets, allowing for a comparative analysis of their respective price movements.
Comparison
'''

prompts_fewshot_template2=f'''
### Instruction:
Given 2 sentences, your task is to classify the relation between them. Your output should be one of the four relations: Temporal, Comparison, Contingency, Expansion. Remember that you only need to output a word.

### Input:
<ARG1>Was this why some of the audience departed before or during the second half?
Or was it because Ms. Collins had gone</ARG1><ARG2>Either way it was a pity</ARG2>

### Response:Expansion

### Instruction:
Given 2 sentences, your task is to classify the relation between them. Your output should be one of the four relations: Temporal, Comparison, Contingency, Expansion. Remember that you only need to output a word.

### Input:
<ARG1>The rest of the concert was more straight jazz and mellow sounds written by Charlie Parker, Ornette Coleman, Bill Douglas and Eddie Gomez, with pictures for the Douglas pieces</ARG1><ARG2>It was enjoyable to hear accomplished jazz without having to sit in a smoke-filled club, but like the first half, much of it was easy to take and ultimately forgettable</ARG2>

### Response:Expansion

### Instruction:
Given 2 sentences, your task is to classify the relation between them. Your output should be one of the four relations: Temporal, Comparison, Contingency, Expansion. Remember that you only need to output a word.

### Input:
<ARG1>Mr. Mayor's hope that references to "press freedom" would survive unamended seems doomed to failure</ARG1><ARG2>the current phrasing is "educating the public and media to avoid manipulation</ARG2>

### Response:Contingency

### Instruction:
Given 2 sentences, your task is to classify the relation between them. Your output should be one of the four relations: Temporal, Comparison, Contingency, Expansion. Remember that you only need to output a word.

### Input:
<ARG1>The Orwellian "New World Information Order" would give government officials rights against the press</ARG1><ARG2>journalists would be obliged to kowtow to their government, which would have licensing and censorship powers and, indeed, duties to block printing of "wrong" ideas</ARG2>

### Response:Contingency

### Instruction:
Given 2 sentences, your task is to classify the relation between them. Your output should be one of the four relations: Temporal, Comparison, Contingency, Expansion. Remember that you only need to output a word.

### Input:
<ARG1>He will be in charge of research, equity sales and trading, and the syndicate operation of Rothschild</ARG1><ARG2>Mr. Conlon was executive vice president and director of the equity division of the international division of Nikko Securities Co</ARG2>

### Response:Temporal

### Instruction:
Given 2 sentences, your task is to classify the relation between them. Your output should be one of the four relations: Temporal, Comparison, Contingency, Expansion. Remember that you only need to output a word.

### Input:
<ARG1>He gave up seven hits, walked five and didn't get a decision</ARG1><ARG2>Arm troubles forced him back to the minors the next year</ARG2>

### Response:Temporal

### Instruction:
Given 2 sentences, your task is to classify the relation between them. Your output should be one of the four relations: Temporal, Comparison, Contingency, Expansion. Remember that you only need to output a word.

### Input:
<ARG1>Seats currently are quoted at $400,000 bid, $425,000 asked</ARG1><ARG2>The record price for a full membership on the exchange is $550,000, set March 9</ARG2>

### Response:Comparison

### Instruction:
Given 2 sentences, your task is to classify the relation between them. Your output should be one of the four relations: Temporal, Comparison, Contingency, Expansion. Remember that you only need to output a word.

### Input:
<ARG1>In late trading, the shares were up a whopping 122 pence ($1.93) -- a 16.3% gain -- to a record 869 pence on very heavy volume of 9.7 million shares</ARG1><ARG2>In the U.S. over-the-counter market, Jaguar shares trading as American Depositary Receipts closed at $13.625, up $1.75</ARG2>

### Response:Comparison
'''

prompts_fewshot_template1=f'''
### Instruction:
Given 2 sentences, your task is to classify the relation between them. Your output should be one of the four relations: Temporal, Comparison, Contingency, Expansion. Remember that you only need to output a word.

### Input:
<ARG1>Was this why some of the audience departed before or during the second half?
Or was it because Ms. Collins had gone</ARG1><ARG2>Either way it was a pity</ARG2>

### Response:Expansion

### Instruction:
Given 2 sentences, your task is to classify the relation between them. Your output should be one of the four relations: Temporal, Comparison, Contingency, Expansion. Remember that you only need to output a word.

### Input:
<ARG1>Mr. Mayor's hope that references to "press freedom" would survive unamended seems doomed to failure</ARG1><ARG2>the current phrasing is "educating the public and media to avoid manipulation</ARG2>

### Response:Contingency

### Instruction:
Given 2 sentences, your task is to classify the relation between them. Your output should be one of the four relations: Temporal, Comparison, Contingency, Expansion. Remember that you only need to output a word.

### Input:
<ARG1>He will be in charge of research, equity sales and trading, and the syndicate operation of Rothschild</ARG1><ARG2>Mr. Conlon was executive vice president and director of the equity division of the international division of Nikko Securities Co</ARG2>

### Response:Temporal

### Instruction:
Given 2 sentences, your task is to classify the relation between them. Your output should be one of the four relations: Temporal, Comparison, Contingency, Expansion. Remember that you only need to output a word.

### Input:
<ARG1>In late trading, the shares were up a whopping 122 pence ($1.93) -- a 16.3% gain -- to a record 869 pence on very heavy volume of 9.7 million shares</ARG1><ARG2>In the U.S. over-the-counter market, Jaguar shares trading as American Depositary Receipts closed at $13.625, up $1.75</ARG2>

### Response:Comparison
'''

