# v13.1 Wave 1 (8B constructor): optimal states, realized prompts, and controls

## Terminology correction

- `M_omega` is the fixed target metric behavior. v13.1 does not optimize an old-style checklist subset `Omega*`.
- `p*` below is the best **observed mined prompt**.
- `s*_g` is the best hypothetical 12-bit state in pool `g` used for the exact structural cap. It may not be realized by any prompt.
- When multiple states tie, the listed state is the smallest integer representative, not a unique optimum.

## Best observed mined prompts

| target | channel | A | source family | prompt summary |
|---|---|---:|---|---|
| 0 | mcq | 0.018764 | humor_R3_metric0\|c21aa4c99f00c83c | **Evaluation Rubric: Ethical Target Selection (Punching Up)** To evaluate the extent to which a text exhibits ethical target selection ("punching up" at those in power or hypocrisy, rather than marginalized/victims), avoiding stereotypes and humiliating shots, and calibrating tone to be teasing, not bullying, follow this rubric: **Overall Score:** 0-3 points 1. **Power Dynamics and Hypocrisy (1 point)**: - 0 points:  |
| 0 | behavioral | 0.018328 | humor_R3_metric50\|611755c2075313a8 | **Rubric for Evaluating "Misdirection and Reveal Design" Metric** 1. **Initial Misdirection:** - *Clarity of the Initial Interpretation:* The text must present an initial scenario or interpretation that is fair and plausible, engaging the reader with a clear expectation. - *Effectiveness of the Misdirection:* The text should convincingly lean the reader towards a specific viewpoint or expectation, without overtly hig |
| 10 | mcq | 0.001119 | humor_R3_metric0\|c21aa4c99f00c83c | ### Rubric for Evaluating "Ethical Target Selection (Punching Up)" **Objective:** The evaluator must determine whether a text adheres to the principles of ethical target selection, focusing on "punching up" rather than targeting marginalized or victim groups, avoiding stereotyping, and ensuring the tone remains teasing rather than bullying. 1. **Target Identification:** - **Primary Criterion:** The target of any sati |
| 10 | behavioral | 0.038613 | humor_R3_metric10\|2d438f35153d3962 | **Rubric for Evaluating "Topical Anchoring and Freshness"** 1. **Selection of Salient Topics**: - The text must center around topics that are relevant and widely recognized by the intended audience. To determine salience, evaluate the potential interest and relatability of the topic to the general audience. An anchored topic should not be obscure or overly niche unless it is presented in a way that makes its relevanc |
| 11 | mcq | 0.016108 | humor_R3_metric12\|87f6b409ad06a01e | Does the text seamlessly integrate current real-world elements with satirical commentary in a manner that a careful reader can easily recognize the satire without misinterpreting the factual information it references? |
| 11 | behavioral | 0.009688 | humor_R3_metric0\|c21aa4c99f00c83c | Does the content specifically target individuals or entities with power or hypocrisy for critique while avoiding significant harm or stereotyping of marginalized groups, and is the tone maintained at a level of teasing rather than bullying? |
| 12 | mcq | 0.010626 | humor_R3_metric0\|c21aa4c99f00c83c | Does the text consistently target figures or entities of power or notable hypocrisy without resorting to stereotypes or humiliating portrayals of marginalized or victimized groups, maintaining a tone that is more teasing than bullying? |
| 12 | behavioral | 0.025103 | humor_R3_metric12\|87f6b409ad06a01e | Does the text incorporate real, verifiable facts as a foundation for its satirical elements while maintaining identifiable stylistic and structural cues that distinguish it clearly as satire? |
| 34 | mcq | 0.000000 | humor_R3_metric0\|c21aa4c99f00c83c | ### Ethical Target Selection (Punching Up) Evaluation Rubric #### Objective: To assess whether a text ethically targets individuals or entities by favoring "punching up" at those in positions of power or those exhibiting hypocrisy, while avoiding harm to marginalized or victimized groups. The text should avoid stereotyping or humiliating individuals and maintain a tone of teasing rather than bullying. #### Evaluation |
| 34 | behavioral | 0.007197 | humor_R3_metric0\|c21aa4c99f00c83c | Does the text target individuals or entities in positions of power or exhibiting hypocrisy for critique, without directing derogatory or humiliating content at marginalized groups or victims? |
| 50 | mcq | 0.000103 | humor_R3_metric0\|c21aa4c99f00c83c | Does the text target individuals or entities with actual power or historical hypocrisy, rather than those who are already marginalized or victims, and does it refrain from using stereotypes or a bullying tone in its criticism? |
| 50 | behavioral | 0.023669 | humor_R3_metric10\|2d438f35153d3962 | Does the text contain jokes that are based on recent events or cultural references, providing just enough context so that they are understandable without being overly explained? |

## Control outcomes

| metric | MCQ q0 | best-expl. | blind MI U / NV | achieved U / NV | constant-vector U / NV | rho | gold fidelity |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.415 | 0.745 | 0.001 / 0.006 | 0.018 / 0.028 | 0.132 / 0.160 | 0.5769 | unavailable |
| 10 | 0.424 | 0.436 | 0.094 / 0.094 | 0.039 / 0.068 | 0.124 / 0.132 | 0.1519 | unavailable |
| 11 | 0.224 | 0.474 | 0.068 / 0.068 | 0.010 / 0.021 | 0.124 / 0.196 | 0.0887 | unavailable |
| 12 | 0.230 | 0.172 | 0.094 / 0.094 | 0.025 / 0.035 | 0.126 / 0.175 | 0.7548 | unavailable |
| 34 | 0.317 | 0.014 | 0.007 / 0.007 | 0.007 / 0.007 | 0.105 / 0.153 | None | unavailable |
| 50 | 0.481 | 0.385 | 0.113 / 0.113 | 0.024 / 0.038 | 0.172 / 0.191 | 0.0783 | unavailable |

## Exact pool states and unit dictionary

Each pool defines local units `u1`...`u12`. A set lists the units labeled 1. Full target labels and probe text follow each state row.

### Metric 0: Assesses who/what is targeted and how: favor punching up at power/hypocrisy over marginalized/victims; avoid stereotyping or humiliating shots; calibrate tone to teasing not bullying.

#### pool_1

| object | 12-bit state | positive unit set | value | tied optima |
|---|---|---|---:|---:|
| target M_omega | `010010000000` | {u2, u5} | — | — |
| MCQ cap state | `001100000001` | {u3, u4, u12} | 0.028028 | 1 |
| behavioral cap state | `000000100010` | {u7, u11} | 0.034942 | 4 |
| no-verbatim cap state | `000010010100` | {u5, u8, u10} | 0.064625 | 8 |
| best observed MCQ prompt | `000000101001` | {u7, u9, u12} | — | — |
| best observed behavioral prompt | `000000001000` | {u9} | — | — |

| unit | probe index | target | probe text |
|---|---:|---:|---|
| u1 | 182 | 0 | Your mom is so fat Her school picture from first grade is still printing |
| u2 | 142 | 1 | The other night I was lying in bed, looking up at the stars, and I wondered, Where the hell is my roof? |
| u3 | 295 | 0 | Husband: Honey why are there broken condoms on the couch? |
| u4 | 192 | 0 | Why did no one laugh when queen farted at the dinner table? Because noble gasses cause no reaction |
| u5 | 290 | 1 | I had a terrible nightmare last night that I ate a muffler. Today, I'm so exhausted. |
| u6 | 60 | 0 | The artist, formerly known as E is now known as MC squared. |
| u7 | 81 | 0 | What do you call a family in which everyone from grand parents to grand kids smoke weed? Joint Family. |
| u8 | 1 | 0 | It's like the weather saw a state trooper It went from 90 to 45 real quick |
| u9 | 246 | 0 | A college girl brings her new beau home to meet the family Her father takes him aside for a chat, "You seem like a nice enough fellow young man, what do you study?' "I'm a theology major sir." Answers the young man. "I see,If you dont mind my asking, where will you live with my little girl after you get married?" "God will provide." "What will you do for a living? How are you going to earn enough to put food on the table?" "God will provide" At the end of the evening, the girl goes up to her father and says " Daddy,I really like him,what do you think?" The father responds,"He seems like a nice enough fellow,but he seems to think I'm god." |
| u10 | 286 | 0 | Why did the lawyer go to culinary school? He wanted to be a sue chef |
| u11 | 99 | 0 | I finally realized I could no longer keep my broken money making machine. It just didn't make cents. |
| u12 | 132 | 0 | "Oh, no", the husband replies. "She's left handed." A wife asks her husband, "Honey, if I died, would you remarry?" oh wait fuck |

#### pool_2

| object | 12-bit state | positive unit set | value | tied optima |
|---|---|---|---:|---:|
| target M_omega | `000110000000` | {u4, u5} | — | — |
| MCQ cap state | `010011001000` | {u2, u5, u6, u9} | 0.038009 | 1 |
| behavioral cap state | `010110000000` | {u2, u4, u5} | 0.030700 | 2 |
| no-verbatim cap state | `010100100000` | {u2, u4, u7} | 0.023843 | 4 |
| best observed MCQ prompt | `010010001100` | {u2, u5, u9, u10} | — | — |
| best observed behavioral prompt | `000110000000` | {u4, u5} | — | — |

| unit | probe index | target | probe text |
|---|---:|---:|---|
| u1 | 144 | 0 | What type of person names their child after a popular hotel chain? |
| u2 | 164 | 0 | What sort of mint do anarchists hate? Governmint |
| u3 | 199 | 0 | Did you know that after this next album, Matisyahu will be retiring? Soon he will be Jah-bless. |
| u4 | 149 | 1 | A man was marooned on a desert island. One day a beautiful woman arrives in a wet suit. 'When did you last have a smoke?' she asks. 'Five years ago.' So she gets out a cigar and he smokes it. She unzips her wet suit a bit and says, 'When did you last have a drink?' He said, 'Five years ago.' So she gets out a bottle of Scotch and he has a drink. Then she unzips her wet suit a bit more and says, 'And when was the last time you played around?' He looks at her in amazement and says: 'You're not telling me you've got a set of golf clubs in there?' Ronnie Corbett (1930 - 2016) |
| u5 | 52 | 1 | A woman gets a call from kidnappers. "We have your son," said the kidnapper. "I don't have a son," says the woman. "Then who just asked for warm milk and made us cut the crust off his sandwiches?" "Oh, God you have my husband!" |
| u6 | 232 | 0 | Importance of meditation When a wife keeps her head on a mans chest and slowly asks "Dear do have any other woman in your life" Remember, the answer is not important at this time Important is - the heartbeat, keep your heart beats in control Therefore, meditate regularly. |
| u7 | 93 | 0 | What does a woman and kentucky fried chicken have in common?? By the time your finished with the breast and thighs, all you left is a greasy box to put your bone in. |
| u8 | 109 | 0 | A man had three horses One was black. One was white. One was eating. |
| u9 | 74 | 0 | A hole was discovered in the fence surrounding the local nudist colony Police are looking into it |
| u10 | 271 | 0 | A compulsive liar walks into a doctor's office claiming to be constipated... The doctor tells him he's full of shit. |
| u11 | 69 | 0 | Timing. Why can't drummers tell jokes? |
| u12 | 194 | 0 | If you set an episode of game of thrones away now, the opening credits will be coming to an end just when the clock strikes midnight on New Year's Eve. Start your year right. |

#### pool_3

| object | 12-bit state | positive unit set | value | tied optima |
|---|---|---|---:|---:|
| target M_omega | `000000100000` | {u7} | — | — |
| MCQ cap state | `010001000101` | {u2, u6, u10, u12} | 0.051725 | 1 |
| behavioral cap state | `000010100000` | {u5, u7} | 0.032119 | 4 |
| no-verbatim cap state | `001010100000` | {u3, u5, u7} | 0.056504 | 4 |
| best observed MCQ prompt | `000000110001` | {u7, u8, u12} | — | — |
| best observed behavioral prompt | `000000100000` | {u7} | — | — |

| unit | probe index | target | probe text |
|---|---:|---:|---|
| u1 | 175 | 0 | Every time I walk into a store with my dad Worker: "Can I help you?" Dad: "No, he was born like that." |
| u2 | 55 | 0 | [Oh, yeah?] My ex-wife cheated on me with a communist! ...there were so many red flags. |
| u3 | 275 | 0 | What do you call a man with 6,022 x 10^23 dollars? A Moleionaire |
| u4 | 201 | 0 | What happens when you put two and two together? A Siamese orgy. |
| u5 | 219 | 0 | Someone asked me if I had ever noticed that I had a keen sense for being able to tell where water was underground... I replied, "I'm well aware." |
| u6 | 235 | 0 | Hay girl you know I am a horny praying mantis? |
| u7 | 294 | 1 | An Irishman walks into a bar and orders three shots of whiskey. The bartender asks him why he ordered three shots. "My life-long friends and I have a tradition. We grew up together but have since gone our separate ways. One is in England and one in the USA, but we each go into a bar on the same day every year and order three shots of whiskey. It's as if we are drinking them together." He then drinks the shots and leaves the bar. The next couple years, he returns and does the same. Then, one year the man returns but only orders two shots. He drinks them both. "I can't help but notice you only ordered two shots," the bartender said. "It appears you must have lost one of your friends. My condolences." "Oh no," the Irishman said. "Those chaps are doing fine. I just quit drinking, that's all." |
| u8 | 40 | 0 | Out of all these emails I'm getting from businesses and companies about what they're doing to prevent the spread of coronavirus.... I'm mostly impressed to hear that our local massage parlor is beginning to use Purell as massage lubricant. I see they're still making money hand over fist. |
| u9 | 257 | 0 | Are you good at math..? ...then give me your number. |
| u10 | 130 | 0 | A Buddhist goes to a hot dog vendor A Buddhist goes to a hot dog vendor, and the vendor asks him "Hey buddy what can I make ya?" "Make me one with everything" EDIT Punctuation |
| u11 | 48 | 0 | 2016 had many celebrity deaths.. ..but 2017 will Hurt the most. :( |
| u12 | 177 | 0 | How many chains does it take to enslave a black person in the twenty-first century? Two Chainz. |

#### pool_4

| object | 12-bit state | positive unit set | value | tied optima |
|---|---|---|---:|---:|
| target M_omega | `000000100000` | {u7} | — | — |
| MCQ cap state | `001010010110` | {u3, u5, u8, u10, u11} | 0.024756 | 1 |
| behavioral cap state | `010000001001` | {u2, u9, u12} | 0.049007 | 8 |
| no-verbatim cap state | `010000011001` | {u2, u8, u9, u12} | 0.064648 | 2 |
| best observed MCQ prompt | `000000010000` | {u8} | — | — |
| best observed behavioral prompt | `000010000000` | {u5} | — | — |

| unit | probe index | target | probe text |
|---|---:|---:|---|
| u1 | 288 | 0 | How did Columbus discover the New World? He sailed there occidentally. |
| u2 | 103 | 0 | I've determined that saying big words always will make you sound smart Totally photosynthesis right? |
| u3 | 123 | 0 | The other day this really fat guy was telling me he ran a marathon last week. But that just doesn't track. |
| u4 | 160 | 0 | I ate horse once by accident... My pulse started racing and my stomach was aching so I called my wife then went to the hospital. When she came to visit, she asked the Doctor how I was. He told her I was in “stable” condition. |
| u5 | 126 | 0 | Fucking Doctors . . . A doctor walks into a nursing surgery. Say's to the mother, "I've got good news and i've got bad news." Mother says, "Give me the bad news." Doctor says, "Your baby was born with red hair" Mother says, "Give me the good news" Doctor says, "It died." |
| u6 | 106 | 0 | Most girls are like modern computers They won’t accept my 3 1/2” floppy |
| u7 | 240 | 1 | Jello has created a product that deters insects. It's very effective, but the flavor is OFF-pudding. |
| u8 | 179 | 0 | Knowledge is Power A jew once went on a walk with his son in a small town and said : -"My dear child, the rabbi told me "If you don't learn anything, you'll become nothing. But if you learn something you will be a well respected person, you will become a rabbi or get a high ranked job... but if you learn nothing, you will become a coachman, a servant or even a soldier."" At this moment a general passed by. The jew saw the general, pointed at him and said : -"You see, my child, thats how you will look!" (This joke was translated from Yiddish, I know it from jewish storyteller Fritz Muliar.) |
| u9 | 100 | 0 | The soft spot on baby's head is basically an off button If you press hard enough |
| u10 | 90 | 0 | What happened to the naughty wire? It was grounded... |
| u11 | 242 | 0 | I talked back to my OCD mother... she was very quick to put me in my place. |
| u12 | 148 | 0 | What does Gordon Ramsey shout when he sees a baby duckling gif on Reddit? IT'S FUCKING r/aww ! |

#### pool_5

| object | 12-bit state | positive unit set | value | tied optima |
|---|---|---|---:|---:|
| target M_omega | `000000010000` | {u8} | — | — |
| MCQ cap state | `111010101110` | {u1, u2, u3, u5, u7, u9, u10, u11} | 0.020268 | 1 |
| behavioral cap state | `010100011010` | {u2, u4, u8, u9, u11} | 0.034509 | 4 |
| no-verbatim cap state | `011100100010` | {u2, u3, u4, u7, u11} | 0.026523 | 2 |
| best observed MCQ prompt | `100010001010` | {u1, u5, u9, u11} | — | — |
| best observed behavioral prompt | `100000100000` | {u1, u7} | — | — |

| unit | probe index | target | probe text |
|---|---:|---:|---|
| u1 | 58 | 0 | Bag Lady A little old lady was walking down the street dragging two large plastic garbage bags behind her. One of the bags was ripped and every once in awhile a $20 bill fell out onto the sidewalk. A policeman stopped her, and said, "Ma'am, there are $20 bills falling out of that bag." "Oh, really? Darn it!" said the little old lady. "I'd better go back and pick them up. Thanks for telling me, Officer." "Not so fast," said the cop. "Where did you get all that money? Did you steal it?" "Oh, no, no", said the old lady. "You see, my back yard is right next to a golf course. A lot of golfers come and pee through a knot hole in my fence, right into my flower garden. It kills the flowers, you know. Then I thought, 'why not make the best of it?' So, now, I stand behind the fence by the knot hole, real quiet, with my hedge clippers. Every time some guy sticks his thing through my fence, I surprise him, grab hold of it and say, 'O.K., buddy! Give me $20 or off it comes!' "Well, that seems only fair," said the cop, laughing. "OK. Good luck! Oh, by the way, what's in the other bag?" "Not everybody pays." |
| u2 | 176 | 0 | Bad news : I farted during my interview for bartender |
| u3 | 29 | 0 | I went to the Doctor and he said that I was paranoid He didn't actually say that but I knew what he was thinking. |
| u4 | 230 | 0 | A eunuch wants to commit suicide. He's told that he doesn't have the balls for this, so he gets prosthetic testicles. |
| u5 | 155 | 0 | How do you cure a heart attack? You give your cardiac a rest. |
| u6 | 264 | 0 | On her deathbed my wife said, "Sweety, I will see you in Heaven." Since then I have kicked a puppy, stole from 4 shops and set fire to an orphanage.! |
| u7 | 268 | 0 | Plane crash Two ladies, one white and one black is flying. Suddenly, the plane hits bad weather and the pilot declares that he is going to crash land. Hearing this, the black lady starts to undress. Surprised, the other ask, "what the hell are you doing? " She replies, "don't you know the first thing they look for after a crash is black box?". |
| u8 | 67 | 1 | Private investigator (At a fancy diner with wife and her friend) Wife's friend: So, what do you do for a living? Me: I'm a private investigator Wife: Kieth, you're allowed to say gynecologist Me: People are eating, Linda! |
| u9 | 23 | 0 | What do you call it when a cow get's milked without consent? "Moo-lestation" |
| u10 | 36 | 0 | Boss: "and this is what you'll be making before taxes" Employee: "that's gross" |
| u11 | 181 | 0 | Two police officers and their dog are walking down the street Two police officers and their dog are walking down the street. One of the officers turns to the other and asks, "Hey partner, how many penises does Buster have?" "Why, one of course" replies the other cop. "I'm not so sure about that, let's just check" says the first policeman. So they look down and sure enough they see only one penis. "That's strange" says the first cop, "this guy at the bar said 'Here goes that dog again with two dicks'" |
| u12 | 84 | 0 | As a kid i looked up to Bill Nye the science guy, but as of today after learning about him i will probably look down on him. |

#### pool_6

| object | 12-bit state | positive unit set | value | tied optima |
|---|---|---|---:|---:|
| target M_omega | `001000000000` | {u3} | — | — |
| MCQ cap state | `010001001000` | {u2, u6, u9} | 0.024988 | 1 |
| behavioral cap state | `100111000001` | {u1, u4, u5, u6, u12} | 0.032816 | 2 |
| no-verbatim cap state | `001000001111` | {u3, u9, u10, u11, u12} | 0.057000 | 2 |
| best observed MCQ prompt | `000000001001` | {u9, u12} | — | — |
| best observed behavioral prompt | `000100011000` | {u4, u8, u9} | — | — |

| unit | probe index | target | probe text |
|---|---:|---:|---|
| u1 | 3 | 0 | Why was the snow yellow? Elsa let it go! |
| u2 | 284 | 0 | Kid : mom, how come you are white and I'm black ? Mom : if I can vaguely remember the things I did in that rave party, just be thankful that you are not barking!!! |
| u3 | 14 | 1 | What is a squirrel's second favorite food? |
| u4 | 184 | 0 | a Country Boy visited the City and met a girl in a Bar who invited him back to her house, When they got there, she undressed and told him to get naked too. She said: "Let's start with a 69" The Country Boy replied: "What's that?" With that she got him into position, and they went at it Within a minute of starting, the City Girl felt a fart coming on She tried holding it back, but she figured the Country Boy was probably enjoying what she was doing to him and just let it rip Less than a Minute later, she felt another one coming on and since he hadn't said anything, let this one out as well After that, the Country Boy pushed her off, got up, and started getting dressed The City Girl, embarrassed, asked "I guess you didn't like that, huh?" The Country Boy said: "No, it was fine, but I just don't think I could take 67 more of those" |
| u5 | 57 | 0 | If Marty McFly had bipolar disorder... ....would that have made him Sweet n' Sour Chicken? |
| u6 | 166 | 0 | Whats the only recoil with the OK BOOMER? |
| u7 | 32 | 0 | How many people does it take to screw in a lightbulb? Two, but good luck finding a lightbulb that big. |
| u8 | 137 | 0 | Three guys die... and Saint Peter stops them at the Golden Gates. He tells them, "Depending how faithful you were to your wife, depends what kind of car you drive across the Golden Bridge to heaven." &amp;#x200B; First guy says, "I was married 10 years and only cheated three times." &amp;#x200B; Saint Peter says, "That's ok I suppose, here take this older model pick-up truck." &amp;#x200B; Second guy says, "I was married 15 years and only cheated once!" &amp;#x200B; Saint Peter says, "Pretty great, here take this sports car." &amp;#x200B; Third guy says, "I was married 40 years and never cheated on my wife." &amp;#x200B; Saint Peter says, "Wow that's the best I've ever heard! Here, take this Golden Edition Rolls-Royce." &amp;#x200B; The three guys start across the bridge and the Rolls takes off and leaves them. About half way across, the other two guys find the Rolls pulled over with his head on the steering wheel. They stop and walk over. &amp;#x200B; First guy says, "Come on man, being dead isn't so bad." &amp;#x200B; Second guy says, "Yeah, look what you're driving, and look what we're driving." &amp;#x200B; Third guy says, "No guys, you don't get it, I just saw my wife go by on a skateboard!" |
| u9 | 248 | 0 | Robert was arrested for stealing an 100 inch TV and a dishwasher from his fiance While in the police car, a policeman turned to him and said "What were you thinking? Were you drunk?" He immediately replied: "No, she asked me to do it!" "How?!", answered the policeman. "We were having a conversation and she said 'Oh please Rob!'" |
| u10 | 269 | 0 | What did the motivational speaker ask the bottle of water? Do you have what it takes to be a liter? |
| u11 | 133 | 0 | I found my wife, my soulmate, my best friend on tinder I guess I wasn't invited to the orgy. |
| u12 | 171 | 0 | What type of yoga does Jesus do? Pontius Pilates. |

### Metric 10: Select salient topics and anchor jokes to clear, current references with sharp, fresh angles and just-enough context for instant intelligibility.

#### pool_1

| object | 12-bit state | positive unit set | value | tied optima |
|---|---|---|---:|---:|
| target M_omega | `010000100100` | {u2, u7, u10} | — | — |
| MCQ cap state | `000000000000` | {empty} | 0.000000 | 4096 |
| behavioral cap state | `010001000101` | {u2, u6, u10, u12} | 0.050049 | 4 |
| no-verbatim cap state | `010101000110` | {u2, u4, u6, u10, u11} | 0.094950 | 4 |
| best observed MCQ prompt | `000101010100` | {u4, u6, u8, u10} | — | — |
| best observed behavioral prompt | `100001100110` | {u1, u6, u7, u10, u11} | — | — |

| unit | probe index | target | probe text |
|---|---:|---:|---|
| u1 | 182 | 0 | Your mom is so fat Her school picture from first grade is still printing |
| u2 | 142 | 1 | The other night I was lying in bed, looking up at the stars, and I wondered, Where the hell is my roof? |
| u3 | 295 | 0 | Husband: Honey why are there broken condoms on the couch? |
| u4 | 290 | 0 | I had a terrible nightmare last night that I ate a muffler. Today, I'm so exhausted. |
| u5 | 60 | 0 | The artist, formerly known as E is now known as MC squared. |
| u6 | 223 | 0 | I have a friend who recently joined a religious yogurt group. He didn't realise it was Ya-Kult. |
| u7 | 81 | 1 | What do you call a family in which everyone from grand parents to grand kids smoke weed? Joint Family. |
| u8 | 14 | 0 | What is a squirrel's second favorite food? |
| u9 | 132 | 0 | "Oh, no", the husband replies. "She's left handed." A wife asks her husband, "Honey, if I died, would you remarry?" oh wait fuck |
| u10 | 149 | 1 | A man was marooned on a desert island. One day a beautiful woman arrives in a wet suit. 'When did you last have a smoke?' she asks. 'Five years ago.' So she gets out a cigar and he smokes it. She unzips her wet suit a bit and says, 'When did you last have a drink?' He said, 'Five years ago.' So she gets out a bottle of Scotch and he has a drink. Then she unzips her wet suit a bit more and says, 'And when was the last time you played around?' He looks at her in amazement and says: 'You're not telling me you've got a set of golf clubs in there?' Ronnie Corbett (1930 - 2016) |
| u11 | 93 | 0 | What does a woman and kentucky fried chicken have in common?? By the time your finished with the breast and thighs, all you left is a greasy box to put your bone in. |
| u12 | 232 | 0 | Importance of meditation When a wife keeps her head on a mans chest and slowly asks "Dear do have any other woman in your life" Remember, the answer is not important at this time Important is - the heartbeat, keep your heart beats in control Therefore, meditate regularly. |

#### pool_2

| object | 12-bit state | positive unit set | value | tied optima |
|---|---|---|---:|---:|
| target M_omega | `000001100100` | {u6, u7, u10} | — | — |
| MCQ cap state | `110001100001` | {u1, u2, u6, u7, u12} | 0.007163 | 8 |
| behavioral cap state | `000001100111` | {u6, u7, u10, u11, u12} | 0.141770 | 4 |
| no-verbatim cap state | `010101110101` | {u2, u4, u6, u7, u8, u10, u12} | 0.111568 | 4 |
| best observed MCQ prompt | `100001100011` | {u1, u6, u7, u11, u12} | — | — |
| best observed behavioral prompt | `001001100111` | {u3, u6, u7, u10, u11, u12} | — | — |

| unit | probe index | target | probe text |
|---|---:|---:|---|
| u1 | 144 | 0 | What type of person names their child after a popular hotel chain? |
| u2 | 130 | 0 | A Buddhist goes to a hot dog vendor A Buddhist goes to a hot dog vendor, and the vendor asks him "Hey buddy what can I make ya?" "Make me one with everything" EDIT Punctuation |
| u3 | 55 | 0 | [Oh, yeah?] My ex-wife cheated on me with a communist! ...there were so many red flags. |
| u4 | 199 | 0 | Did you know that after this next album, Matisyahu will be retiring? Soon he will be Jah-bless. |
| u5 | 201 | 0 | What happens when you put two and two together? A Siamese orgy. |
| u6 | 52 | 1 | A woman gets a call from kidnappers. "We have your son," said the kidnapper. "I don't have a son," says the woman. "Then who just asked for warm milk and made us cut the crust off his sandwiches?" "Oh, God you have my husband!" |
| u7 | 294 | 1 | An Irishman walks into a bar and orders three shots of whiskey. The bartender asks him why he ordered three shots. "My life-long friends and I have a tradition. We grew up together but have since gone our separate ways. One is in England and one in the USA, but we each go into a bar on the same day every year and order three shots of whiskey. It's as if we are drinking them together." He then drinks the shots and leaves the bar. The next couple years, he returns and does the same. Then, one year the man returns but only orders two shots. He drinks them both. "I can't help but notice you only ordered two shots," the bartender said. "It appears you must have lost one of your friends. My condolences." "Oh no," the Irishman said. "Those chaps are doing fine. I just quit drinking, that's all." |
| u8 | 109 | 0 | A man had three horses One was black. One was white. One was eating. |
| u9 | 74 | 0 | A hole was discovered in the fence surrounding the local nudist colony Police are looking into it |
| u10 | 271 | 1 | A compulsive liar walks into a doctor's office claiming to be constipated... The doctor tells him he's full of shit. |
| u11 | 69 | 0 | Timing. Why can't drummers tell jokes? |
| u12 | 194 | 0 | If you set an episode of game of thrones away now, the opening credits will be coming to an end just when the clock strikes midnight on New Year's Eve. Start your year right. |

#### pool_3

| object | 12-bit state | positive unit set | value | tied optima |
|---|---|---|---:|---:|
| target M_omega | `001100000001` | {u3, u4, u12} | — | — |
| MCQ cap state | `000000000000` | {empty} | 0.000000 | 4096 |
| behavioral cap state | `110000101101` | {u1, u2, u7, u9, u10, u12} | 0.044456 | 4 |
| no-verbatim cap state | `111000010011` | {u1, u2, u3, u8, u11, u12} | 0.079197 | 2 |
| best observed MCQ prompt | `000000000001` | {u12} | — | — |
| best observed behavioral prompt | `011100010001` | {u2, u3, u4, u8, u12} | — | — |

| unit | probe index | target | probe text |
|---|---:|---:|---|
| u1 | 175 | 0 | Every time I walk into a store with my dad Worker: "Can I help you?" Dad: "No, he was born like that." |
| u2 | 106 | 0 | Most girls are like modern computers They won’t accept my 3 1/2” floppy |
| u3 | 160 | 1 | I ate horse once by accident... My pulse started racing and my stomach was aching so I called my wife then went to the hospital. When she came to visit, she asked the Doctor how I was. He told her I was in “stable” condition. |
| u4 | 148 | 1 | What does Gordon Ramsey shout when he sees a baby duckling gif on Reddit? IT'S FUCKING r/aww ! |
| u5 | 275 | 0 | What do you call a man with 6,022 x 10^23 dollars? A Moleionaire |
| u6 | 219 | 0 | Someone asked me if I had ever noticed that I had a keen sense for being able to tell where water was underground... I replied, "I'm well aware." |
| u7 | 235 | 0 | Hay girl you know I am a horny praying mantis? |
| u8 | 40 | 0 | Out of all these emails I'm getting from businesses and companies about what they're doing to prevent the spread of coronavirus.... I'm mostly impressed to hear that our local massage parlor is beginning to use Purell as massage lubricant. I see they're still making money hand over fist. |
| u9 | 242 | 0 | I talked back to my OCD mother... she was very quick to put me in my place. |
| u10 | 257 | 0 | Are you good at math..? ...then give me your number. |
| u11 | 48 | 0 | 2016 had many celebrity deaths.. ..but 2017 will Hurt the most. :( |
| u12 | 99 | 1 | I finally realized I could no longer keep my broken money making machine. It just didn't make cents. |

#### pool_4

| object | 12-bit state | positive unit set | value | tied optima |
|---|---|---|---:|---:|
| target M_omega | `110001000000` | {u1, u2, u6} | — | — |
| MCQ cap state | `000000000000` | {empty} | 0.000000 | 4096 |
| behavioral cap state | `100111001001` | {u1, u4, u5, u6, u9, u12} | 0.065885 | 8 |
| no-verbatim cap state | `100111001001` | {u1, u4, u5, u6, u9, u12} | 0.101499 | 4 |
| best observed MCQ prompt | `000000000100` | {u10} | — | — |
| best observed behavioral prompt | `100001110100` | {u1, u6, u7, u8, u10} | — | — |

| unit | probe index | target | probe text |
|---|---:|---:|---|
| u1 | 133 | 1 | I found my wife, my soulmate, my best friend on tinder I guess I wasn't invited to the orgy. |
| u2 | 177 | 1 | How many chains does it take to enslave a black person in the twenty-first century? Two Chainz. |
| u3 | 288 | 0 | How did Columbus discover the New World? He sailed there occidentally. |
| u4 | 103 | 0 | I've determined that saying big words always will make you sound smart Totally photosynthesis right? |
| u5 | 123 | 0 | The other day this really fat guy was telling me he ran a marathon last week. But that just doesn't track. |
| u6 | 67 | 1 | Private investigator (At a fancy diner with wife and her friend) Wife's friend: So, what do you do for a living? Me: I'm a private investigator Wife: Kieth, you're allowed to say gynecologist Me: People are eating, Linda! |
| u7 | 126 | 0 | Fucking Doctors . . . A doctor walks into a nursing surgery. Say's to the mother, "I've got good news and i've got bad news." Mother says, "Give me the bad news." Doctor says, "Your baby was born with red hair" Mother says, "Give me the good news" Doctor says, "It died." |
| u8 | 179 | 0 | Knowledge is Power A jew once went on a walk with his son in a small town and said : -"My dear child, the rabbi told me "If you don't learn anything, you'll become nothing. But if you learn something you will be a well respected person, you will become a rabbi or get a high ranked job... but if you learn nothing, you will become a coachman, a servant or even a soldier."" At this moment a general passed by. The jew saw the general, pointed at him and said : -"You see, my child, thats how you will look!" (This joke was translated from Yiddish, I know it from jewish storyteller Fritz Muliar.) |
| u9 | 100 | 0 | The soft spot on baby's head is basically an off button If you press hard enough |
| u10 | 155 | 0 | How do you cure a heart attack? You give your cardiac a rest. |
| u11 | 90 | 0 | What happened to the naughty wire? It was grounded... |
| u12 | 102 | 0 | A Racist Joke Two racists are sitting on a park bench, watching and making commentary on passersby. "You know," the first racist says, "we're the real victims in today's society." The second racist nods knowingly. "As soon as someone hears that you have a problem with one group of people or another," the first racist continues, "they make all kinds of unfair and unfounded assumptions about you." Once again, the second racist nods. "Just as an example," says the first racist, "if someone were to read a transcript of our conversation, they'd almost certainly think that both of us were white." The second racist grins and says "哈哈！我是公交车司机!" |

#### pool_5

| object | 12-bit state | positive unit set | value | tied optima |
|---|---|---|---:|---:|
| target M_omega | `010010000000` | {u2, u5} | — | — |
| MCQ cap state | `000000000000` | {empty} | 0.000000 | 4096 |
| behavioral cap state | `011101000111` | {u2, u3, u4, u6, u10, u11, u12} | 0.105674 | 4 |
| no-verbatim cap state | `010010001110` | {u2, u5, u9, u10, u11} | 0.123518 | 4 |
| best observed MCQ prompt | `101000100000` | {u1, u3, u7} | — | — |
| best observed behavioral prompt | `110100000100` | {u1, u2, u4, u10} | — | — |

| unit | probe index | target | probe text |
|---|---:|---:|---|
| u1 | 174 | 0 | i poured root beer in a square glass... now i just have beer. |
| u2 | 248 | 1 | Robert was arrested for stealing an 100 inch TV and a dishwasher from his fiance While in the police car, a policeman turned to him and said "What were you thinking? Were you drunk?" He immediately replied: "No, she asked me to do it!" "How?!", answered the policeman. "We were having a conversation and she said 'Oh please Rob!'" |
| u3 | 266 | 0 | For my 2017. New years resolution I'll go 3840 × 2160 |
| u4 | 3 | 0 | Why was the snow yellow? Elsa let it go! |
| u5 | 286 | 1 | Why did the lawyer go to culinary school? He wanted to be a sue chef |
| u6 | 176 | 0 | Bad news : I farted during my interview for bartender |
| u7 | 240 | 0 | Jello has created a product that deters insects. It's very effective, but the flavor is OFF-pudding. |
| u8 | 29 | 0 | I went to the Doctor and he said that I was paranoid He didn't actually say that but I knew what he was thinking. |
| u9 | 230 | 0 | A eunuch wants to commit suicide. He's told that he doesn't have the balls for this, so he gets prosthetic testicles. |
| u10 | 268 | 0 | Plane crash Two ladies, one white and one black is flying. Suddenly, the plane hits bad weather and the pilot declares that he is going to crash land. Hearing this, the black lady starts to undress. Surprised, the other ask, "what the hell are you doing? " She replies, "don't you know the first thing they look for after a crash is black box?". |
| u11 | 181 | 0 | Two police officers and their dog are walking down the street Two police officers and their dog are walking down the street. One of the officers turns to the other and asks, "Hey partner, how many penises does Buster have?" "Why, one of course" replies the other cop. "I'm not so sure about that, let's just check" says the first policeman. So they look down and sure enough they see only one penis. "That's strange" says the first cop, "this guy at the bar said 'Here goes that dog again with two dicks'" |
| u12 | 84 | 0 | As a kid i looked up to Bill Nye the science guy, but as of today after learning about him i will probably look down on him. |

#### pool_6

| object | 12-bit state | positive unit set | value | tied optima |
|---|---|---|---:|---:|
| target M_omega | `000000100100` | {u7, u10} | — | — |
| MCQ cap state | `000110101101` | {u4, u5, u7, u9, u10, u12} | 0.000064 | 16 |
| behavioral cap state | `001010000110` | {u3, u5, u10, u11} | 0.110167 | 4 |
| no-verbatim cap state | `000000000101` | {u10, u12} | 0.146725 | 4 |
| best observed MCQ prompt | `000000000000` | {empty} | — | — |
| best observed behavioral prompt | `100000101111` | {u1, u7, u9, u10, u11, u12} | — | — |

| unit | probe index | target | probe text |
|---|---:|---:|---|
| u1 | 284 | 0 | Kid : mom, how come you are white and I'm black ? Mom : if I can vaguely remember the things I did in that rave party, just be thankful that you are not barking!!! |
| u2 | 57 | 0 | If Marty McFly had bipolar disorder... ....would that have made him Sweet n' Sour Chicken? |
| u3 | 24 | 0 | Found the moron that doesn't know what "thou" means. It's obviously you. |
| u4 | 72 | 0 | Everyone tells me I'm average... That's just mean. |
| u5 | 166 | 0 | Whats the only recoil with the OK BOOMER? |
| u6 | 63 | 0 | Boy catches a priest masturbating and asks, "What are you doing father?" |
| u7 | 32 | 1 | How many people does it take to screw in a lightbulb? Two, but good luck finding a lightbulb that big. |
| u8 | 276 | 0 | What do whores and sailors have in common? They are both always surrounded by sea - men! I made up this one. |
| u9 | 137 | 0 | Three guys die... and Saint Peter stops them at the Golden Gates. He tells them, "Depending how faithful you were to your wife, depends what kind of car you drive across the Golden Bridge to heaven." &amp;#x200B; First guy says, "I was married 10 years and only cheated three times." &amp;#x200B; Saint Peter says, "That's ok I suppose, here take this older model pick-up truck." &amp;#x200B; Second guy says, "I was married 15 years and only cheated once!" &amp;#x200B; Saint Peter says, "Pretty great, here take this sports car." &amp;#x200B; Third guy says, "I was married 40 years and never cheated on my wife." &amp;#x200B; Saint Peter says, "Wow that's the best I've ever heard! Here, take this Golden Edition Rolls-Royce." &amp;#x200B; The three guys start across the bridge and the Rolls takes off and leaves them. About half way across, the other two guys find the Rolls pulled over with his head on the steering wheel. They stop and walk over. &amp;#x200B; First guy says, "Come on man, being dead isn't so bad." &amp;#x200B; Second guy says, "Yeah, look what you're driving, and look what we're driving." &amp;#x200B; Third guy says, "No guys, you don't get it, I just saw my wife go by on a skateboard!" |
| u10 | 58 | 1 | Bag Lady A little old lady was walking down the street dragging two large plastic garbage bags behind her. One of the bags was ripped and every once in awhile a $20 bill fell out onto the sidewalk. A policeman stopped her, and said, "Ma'am, there are $20 bills falling out of that bag." "Oh, really? Darn it!" said the little old lady. "I'd better go back and pick them up. Thanks for telling me, Officer." "Not so fast," said the cop. "Where did you get all that money? Did you steal it?" "Oh, no, no", said the old lady. "You see, my back yard is right next to a golf course. A lot of golfers come and pee through a knot hole in my fence, right into my flower garden. It kills the flowers, you know. Then I thought, 'why not make the best of it?' So, now, I stand behind the fence by the knot hole, real quiet, with my hedge clippers. Every time some guy sticks his thing through my fence, I surprise him, grab hold of it and say, 'O.K., buddy! Give me $20 or off it comes!' "Well, that seems only fair," said the cop, laughing. "OK. Good luck! Oh, by the way, what's in the other bag?" "Not everybody pays." |
| u11 | 171 | 0 | What type of yoga does Jesus do? Pontius Pilates. |
| u12 | 64 | 0 | A tourist in Hawaii is amazed at how healthy and invigorated he feels after just a few days into visiting the islands... He strikes up a conversation with one of the locals while they are wading out into the crystal clear, warm surf on yet another perfect island day. "I just cant get over how beautiful this place is," the tourist says excitedly, "I feel great! I haven't felt this young and healthy in years! Island life is fantastic!" The local says, "I know what you mean! Take me for instance. When I came here I was totally bald, didn't have any teeth and I couldn't even walk...and look at me now!" The tourist looks at him and says, "Wow, that's amazing! How long have you been here?" And the local says, "Oh, I was born here." |

### Metric 11: Convincingly mimic source styles/voices and specifics, balance affectionate vs hostile tone, and deploy the imitation to deliver clear, purposeful comedic commentary.

#### pool_1

| object | 12-bit state | positive unit set | value | tied optima |
|---|---|---|---:|---:|
| target M_omega | `000100000001` | {u4, u12} | — | — |
| MCQ cap state | `001001101000` | {u3, u6, u7, u9} | 0.019202 | 1 |
| behavioral cap state | `010000100000` | {u2, u7} | 0.018399 | 64 |
| no-verbatim cap state | `010101000011` | {u2, u4, u6, u11, u12} | 0.023788 | 4 |
| best observed MCQ prompt | `000100000001` | {u4, u12} | — | — |
| best observed behavioral prompt | `010110101011` | {u2, u4, u5, u7, u9, u11, u12} | — | — |

| unit | probe index | target | probe text |
|---|---:|---:|---|
| u1 | 182 | 0 | Your mom is so fat Her school picture from first grade is still printing |
| u2 | 142 | 0 | The other night I was lying in bed, looking up at the stars, and I wondered, Where the hell is my roof? |
| u3 | 295 | 0 | Husband: Honey why are there broken condoms on the couch? |
| u4 | 52 | 1 | A woman gets a call from kidnappers. "We have your son," said the kidnapper. "I don't have a son," says the woman. "Then who just asked for warm milk and made us cut the crust off his sandwiches?" "Oh, God you have my husband!" |
| u5 | 290 | 0 | I had a terrible nightmare last night that I ate a muffler. Today, I'm so exhausted. |
| u6 | 60 | 0 | The artist, formerly known as E is now known as MC squared. |
| u7 | 223 | 0 | I have a friend who recently joined a religious yogurt group. He didn't realise it was Ya-Kult. |
| u8 | 81 | 0 | What do you call a family in which everyone from grand parents to grand kids smoke weed? Joint Family. |
| u9 | 286 | 0 | Why did the lawyer go to culinary school? He wanted to be a sue chef |
| u10 | 14 | 0 | What is a squirrel's second favorite food? |
| u11 | 132 | 0 | "Oh, no", the husband replies. "She's left handed." A wife asks her husband, "Honey, if I died, would you remarry?" oh wait fuck |
| u12 | 149 | 1 | A man was marooned on a desert island. One day a beautiful woman arrives in a wet suit. 'When did you last have a smoke?' she asks. 'Five years ago.' So she gets out a cigar and he smokes it. She unzips her wet suit a bit and says, 'When did you last have a drink?' He said, 'Five years ago.' So she gets out a bottle of Scotch and he has a drink. Then she unzips her wet suit a bit more and says, 'And when was the last time you played around?' He looks at her in amazement and says: 'You're not telling me you've got a set of golf clubs in there?' Ronnie Corbett (1930 - 2016) |

#### pool_2

| object | 12-bit state | positive unit set | value | tied optima |
|---|---|---|---:|---:|
| target M_omega | `000001000100` | {u6, u10} | — | — |
| MCQ cap state | `011111001100` | {u2, u3, u4, u5, u6, u9, u10} | 0.018326 | 1 |
| behavioral cap state | `000011010001` | {u5, u6, u8, u12} | 0.053218 | 4 |
| no-verbatim cap state | `000011110000` | {u5, u6, u7, u8} | 0.057354 | 8 |
| best observed MCQ prompt | `000101001100` | {u4, u6, u9, u10} | — | — |
| best observed behavioral prompt | `100101011011` | {u1, u4, u6, u8, u9, u11, u12} | — | — |

| unit | probe index | target | probe text |
|---|---:|---:|---|
| u1 | 1 | 0 | It's like the weather saw a state trooper It went from 90 to 45 real quick |
| u2 | 164 | 0 | What sort of mint do anarchists hate? Governmint |
| u3 | 199 | 0 | Did you know that after this next album, Matisyahu will be retiring? Soon he will be Jah-bless. |
| u4 | 99 | 0 | I finally realized I could no longer keep my broken money making machine. It just didn't make cents. |
| u5 | 232 | 0 | Importance of meditation When a wife keeps her head on a mans chest and slowly asks "Dear do have any other woman in your life" Remember, the answer is not important at this time Important is - the heartbeat, keep your heart beats in control Therefore, meditate regularly. |
| u6 | 294 | 1 | An Irishman walks into a bar and orders three shots of whiskey. The bartender asks him why he ordered three shots. "My life-long friends and I have a tradition. We grew up together but have since gone our separate ways. One is in England and one in the USA, but we each go into a bar on the same day every year and order three shots of whiskey. It's as if we are drinking them together." He then drinks the shots and leaves the bar. The next couple years, he returns and does the same. Then, one year the man returns but only orders two shots. He drinks them both. "I can't help but notice you only ordered two shots," the bartender said. "It appears you must have lost one of your friends. My condolences." "Oh no," the Irishman said. "Those chaps are doing fine. I just quit drinking, that's all." |
| u7 | 93 | 0 | What does a woman and kentucky fried chicken have in common?? By the time your finished with the breast and thighs, all you left is a greasy box to put your bone in. |
| u8 | 109 | 0 | A man had three horses One was black. One was white. One was eating. |
| u9 | 74 | 0 | A hole was discovered in the fence surrounding the local nudist colony Police are looking into it |
| u10 | 271 | 1 | A compulsive liar walks into a doctor's office claiming to be constipated... The doctor tells him he's full of shit. |
| u11 | 69 | 0 | Timing. Why can't drummers tell jokes? |
| u12 | 194 | 0 | If you set an episode of game of thrones away now, the opening credits will be coming to an end just when the clock strikes midnight on New Year's Eve. Start your year right. |

#### pool_3

| object | 12-bit state | positive unit set | value | tied optima |
|---|---|---|---:|---:|
| target M_omega | `001000001000` | {u3, u9} | — | — |
| MCQ cap state | `000100001010` | {u4, u9, u11} | 0.042958 | 1 |
| behavioral cap state | `001000100000` | {u3, u7} | 0.029364 | 64 |
| no-verbatim cap state | `001000011000` | {u3, u8, u9} | 0.067532 | 8 |
| best observed MCQ prompt | `001000001000` | {u3, u9} | — | — |
| best observed behavioral prompt | `111010100010` | {u1, u2, u3, u5, u7, u11} | — | — |

| unit | probe index | target | probe text |
|---|---:|---:|---|
| u1 | 144 | 0 | What type of person names their child after a popular hotel chain? |
| u2 | 175 | 0 | Every time I walk into a store with my dad Worker: "Can I help you?" Dad: "No, he was born like that." |
| u3 | 160 | 1 | I ate horse once by accident... My pulse started racing and my stomach was aching so I called my wife then went to the hospital. When she came to visit, she asked the Doctor how I was. He told her I was in “stable” condition. |
| u4 | 55 | 0 | [Oh, yeah?] My ex-wife cheated on me with a communist! ...there were so many red flags. |
| u5 | 275 | 0 | What do you call a man with 6,022 x 10^23 dollars? A Moleionaire |
| u6 | 201 | 0 | What happens when you put two and two together? A Siamese orgy. |
| u7 | 219 | 0 | Someone asked me if I had ever noticed that I had a keen sense for being able to tell where water was underground... I replied, "I'm well aware." |
| u8 | 235 | 0 | Hay girl you know I am a horny praying mantis? |
| u9 | 40 | 1 | Out of all these emails I'm getting from businesses and companies about what they're doing to prevent the spread of coronavirus.... I'm mostly impressed to hear that our local massage parlor is beginning to use Purell as massage lubricant. I see they're still making money hand over fist. |
| u10 | 257 | 0 | Are you good at math..? ...then give me your number. |
| u11 | 130 | 0 | A Buddhist goes to a hot dog vendor A Buddhist goes to a hot dog vendor, and the vendor asks him "Hey buddy what can I make ya?" "Make me one with everything" EDIT Punctuation |
| u12 | 48 | 0 | 2016 had many celebrity deaths.. ..but 2017 will Hurt the most. :( |

#### pool_4

| object | 12-bit state | positive unit set | value | tied optima |
|---|---|---|---:|---:|
| target M_omega | `100000000000` | {u1} | — | — |
| MCQ cap state | `100010000010` | {u1, u5, u11} | 0.041205 | 1 |
| behavioral cap state | `010100000100` | {u2, u4, u10} | 0.017314 | 8 |
| no-verbatim cap state | `000011010010` | {u5, u6, u8, u11} | 0.025638 | 4 |
| best observed MCQ prompt | `100001000010` | {u1, u6, u11} | — | — |
| best observed behavioral prompt | `001000110011` | {u3, u7, u8, u11, u12} | — | — |

| unit | probe index | target | probe text |
|---|---:|---:|---|
| u1 | 177 | 1 | How many chains does it take to enslave a black person in the twenty-first century? Two Chainz. |
| u2 | 288 | 0 | How did Columbus discover the New World? He sailed there occidentally. |
| u3 | 103 | 0 | I've determined that saying big words always will make you sound smart Totally photosynthesis right? |
| u4 | 123 | 0 | The other day this really fat guy was telling me he ran a marathon last week. But that just doesn't track. |
| u5 | 126 | 0 | Fucking Doctors . . . A doctor walks into a nursing surgery. Say's to the mother, "I've got good news and i've got bad news." Mother says, "Give me the bad news." Doctor says, "Your baby was born with red hair" Mother says, "Give me the good news" Doctor says, "It died." |
| u6 | 106 | 0 | Most girls are like modern computers They won’t accept my 3 1/2” floppy |
| u7 | 100 | 0 | The soft spot on baby's head is basically an off button If you press hard enough |
| u8 | 90 | 0 | What happened to the naughty wire? It was grounded... |
| u9 | 242 | 0 | I talked back to my OCD mother... she was very quick to put me in my place. |
| u10 | 148 | 0 | What does Gordon Ramsey shout when he sees a baby duckling gif on Reddit? IT'S FUCKING r/aww ! |
| u11 | 102 | 0 | A Racist Joke Two racists are sitting on a park bench, watching and making commentary on passersby. "You know," the first racist says, "we're the real victims in today's society." The second racist nods knowingly. "As soon as someone hears that you have a problem with one group of people or another," the first racist continues, "they make all kinds of unfair and unfounded assumptions about you." Once again, the second racist nods. "Just as an example," says the first racist, "if someone were to read a transcript of our conversation, they'd almost certainly think that both of us were white." The second racist grins and says "哈哈！我是公交车司机!" |
| u12 | 255 | 0 | A renowned marine biologist is invited to a Great White conservation center The scientists there are very concerned that the Great Whites native to that area had all become sick and had developed digestive issues. The initial thinking was ocean pollution was to blame. The sharks weren't properly processing their meals and seemed to be evacuating their bowels at a much higher than normal frequency. Initially it seemed like it was only the infants that had been impacted but soon the adults seemed to be facing the issue as well. The Marine biologist is skeptical that pollution could cause such widespread issues so after some investigating he is able to figure out that there is actually a virus to blame that seems to be highly contagious and spreads through fecal matter. In other words, Baby shark do doodoo mommy shark do doodoo, daddy shark do doodoo... |

#### pool_5

| object | 12-bit state | positive unit set | value | tied optima |
|---|---|---|---:|---:|
| target M_omega | `000000001000` | {u9} | — | — |
| MCQ cap state | `000010001010` | {u5, u9, u11} | 0.032853 | 1 |
| behavioral cap state | `100001000000` | {u1, u6} | 0.009173 | 16 |
| no-verbatim cap state | `000110011000` | {u4, u5, u8, u9} | 0.020111 | 32 |
| best observed MCQ prompt | `000001001000` | {u6, u9} | — | — |
| best observed behavioral prompt | `101100101001` | {u1, u3, u4, u7, u9, u12} | — | — |

| unit | probe index | target | probe text |
|---|---:|---:|---|
| u1 | 174 | 0 | i poured root beer in a square glass... now i just have beer. |
| u2 | 266 | 0 | For my 2017. New years resolution I'll go 3840 × 2160 |
| u3 | 176 | 0 | Bad news : I farted during my interview for bartender |
| u4 | 240 | 0 | Jello has created a product that deters insects. It's very effective, but the flavor is OFF-pudding. |
| u5 | 29 | 0 | I went to the Doctor and he said that I was paranoid He didn't actually say that but I knew what he was thinking. |
| u6 | 230 | 0 | A eunuch wants to commit suicide. He's told that he doesn't have the balls for this, so he gets prosthetic testicles. |
| u7 | 155 | 0 | How do you cure a heart attack? You give your cardiac a rest. |
| u8 | 268 | 0 | Plane crash Two ladies, one white and one black is flying. Suddenly, the plane hits bad weather and the pilot declares that he is going to crash land. Hearing this, the black lady starts to undress. Surprised, the other ask, "what the hell are you doing? " She replies, "don't you know the first thing they look for after a crash is black box?". |
| u9 | 67 | 1 | Private investigator (At a fancy diner with wife and her friend) Wife's friend: So, what do you do for a living? Me: I'm a private investigator Wife: Kieth, you're allowed to say gynecologist Me: People are eating, Linda! |
| u10 | 23 | 0 | What do you call it when a cow get's milked without consent? "Moo-lestation" |
| u11 | 84 | 0 | As a kid i looked up to Bill Nye the science guy, but as of today after learning about him i will probably look down on him. |
| u12 | 9 | 0 | What rock group has 4 men that do not sing? Mt. Rushmore |

#### pool_6

| object | 12-bit state | positive unit set | value | tied optima |
|---|---|---|---:|---:|
| target M_omega | `000000000010` | {u11} | — | — |
| MCQ cap state | `101010000011` | {u1, u3, u5, u11, u12} | 0.028133 | 1 |
| behavioral cap state | `010010000010` | {u2, u5, u11} | 0.032340 | 64 |
| no-verbatim cap state | `101110000110` | {u1, u3, u4, u5, u10, u11} | 0.039231 | 2 |
| best observed MCQ prompt | `100000110010` | {u1, u7, u8, u11} | — | — |
| best observed behavioral prompt | `010000101111` | {u2, u7, u9, u10, u11, u12} | — | — |

| unit | probe index | target | probe text |
|---|---:|---:|---|
| u1 | 181 | 0 | Two police officers and their dog are walking down the street Two police officers and their dog are walking down the street. One of the officers turns to the other and asks, "Hey partner, how many penises does Buster have?" "Why, one of course" replies the other cop. "I'm not so sure about that, let's just check" says the first policeman. So they look down and sure enough they see only one penis. "That's strange" says the first cop, "this guy at the bar said 'Here goes that dog again with two dicks'" |
| u2 | 3 | 0 | Why was the snow yellow? Elsa let it go! |
| u3 | 284 | 0 | Kid : mom, how come you are white and I'm black ? Mom : if I can vaguely remember the things I did in that rave party, just be thankful that you are not barking!!! |
| u4 | 184 | 0 | a Country Boy visited the City and met a girl in a Bar who invited him back to her house, When they got there, she undressed and told him to get naked too. She said: "Let's start with a 69" The Country Boy replied: "What's that?" With that she got him into position, and they went at it Within a minute of starting, the City Girl felt a fart coming on She tried holding it back, but she figured the Country Boy was probably enjoying what she was doing to him and just let it rip Less than a Minute later, she felt another one coming on and since he hadn't said anything, let this one out as well After that, the Country Boy pushed her off, got up, and started getting dressed The City Girl, embarrassed, asked "I guess you didn't like that, huh?" The Country Boy said: "No, it was fine, but I just don't think I could take 67 more of those" |
| u5 | 57 | 0 | If Marty McFly had bipolar disorder... ....would that have made him Sweet n' Sour Chicken? |
| u6 | 166 | 0 | Whats the only recoil with the OK BOOMER? |
| u7 | 32 | 0 | How many people does it take to screw in a lightbulb? Two, but good luck finding a lightbulb that big. |
| u8 | 137 | 0 | Three guys die... and Saint Peter stops them at the Golden Gates. He tells them, "Depending how faithful you were to your wife, depends what kind of car you drive across the Golden Bridge to heaven." &amp;#x200B; First guy says, "I was married 10 years and only cheated three times." &amp;#x200B; Saint Peter says, "That's ok I suppose, here take this older model pick-up truck." &amp;#x200B; Second guy says, "I was married 15 years and only cheated once!" &amp;#x200B; Saint Peter says, "Pretty great, here take this sports car." &amp;#x200B; Third guy says, "I was married 40 years and never cheated on my wife." &amp;#x200B; Saint Peter says, "Wow that's the best I've ever heard! Here, take this Golden Edition Rolls-Royce." &amp;#x200B; The three guys start across the bridge and the Rolls takes off and leaves them. About half way across, the other two guys find the Rolls pulled over with his head on the steering wheel. They stop and walk over. &amp;#x200B; First guy says, "Come on man, being dead isn't so bad." &amp;#x200B; Second guy says, "Yeah, look what you're driving, and look what we're driving." &amp;#x200B; Third guy says, "No guys, you don't get it, I just saw my wife go by on a skateboard!" |
| u9 | 36 | 0 | Boss: "and this is what you'll be making before taxes" Employee: "that's gross" |
| u10 | 269 | 0 | What did the motivational speaker ask the bottle of water? Do you have what it takes to be a liter? |
| u11 | 133 | 1 | I found my wife, my soulmate, my best friend on tinder I guess I wasn't invited to the orgy. |
| u12 | 171 | 0 | What type of yoga does Jesus do? Pontius Pilates. |

### Metric 12: Mimic news/panel formats credibly while clearly signaling satire; ground bits in real premises, move fast, and define targets so neutral rhetoric heightens absurdity without confusion.

#### pool_1

| object | 12-bit state | positive unit set | value | tied optima |
|---|---|---|---:|---:|
| target M_omega | `001000000101` | {u3, u10, u12} | — | — |
| MCQ cap state | `001001011000` | {u3, u6, u8, u9} | 0.005132 | 2 |
| behavioral cap state | `100110100111` | {u1, u4, u5, u7, u10, u11, u12} | 0.040854 | 8 |
| no-verbatim cap state | `011010100110` | {u2, u3, u5, u7, u10, u11} | 0.055911 | 16 |
| best observed MCQ prompt | `101101111011` | {u1, u3, u4, u6, u7, u8, u9, u11, u12} | — | — |
| best observed behavioral prompt | `001000100101` | {u3, u7, u10, u12} | — | — |

| unit | probe index | target | probe text |
|---|---:|---:|---|
| u1 | 142 | 0 | The other night I was lying in bed, looking up at the stars, and I wondered, Where the hell is my roof? |
| u2 | 295 | 0 | Husband: Honey why are there broken condoms on the couch? |
| u3 | 52 | 1 | A woman gets a call from kidnappers. "We have your son," said the kidnapper. "I don't have a son," says the woman. "Then who just asked for warm milk and made us cut the crust off his sandwiches?" "Oh, God you have my husband!" |
| u4 | 290 | 0 | I had a terrible nightmare last night that I ate a muffler. Today, I'm so exhausted. |
| u5 | 60 | 0 | The artist, formerly known as E is now known as MC squared. |
| u6 | 223 | 0 | I have a friend who recently joined a religious yogurt group. He didn't realise it was Ya-Kult. |
| u7 | 81 | 0 | What do you call a family in which everyone from grand parents to grand kids smoke weed? Joint Family. |
| u8 | 286 | 0 | Why did the lawyer go to culinary school? He wanted to be a sue chef |
| u9 | 14 | 0 | What is a squirrel's second favorite food? |
| u10 | 271 | 1 | A compulsive liar walks into a doctor's office claiming to be constipated... The doctor tells him he's full of shit. |
| u11 | 132 | 0 | "Oh, no", the husband replies. "She's left handed." A wife asks her husband, "Honey, if I died, would you remarry?" oh wait fuck |
| u12 | 149 | 1 | A man was marooned on a desert island. One day a beautiful woman arrives in a wet suit. 'When did you last have a smoke?' she asks. 'Five years ago.' So she gets out a cigar and he smokes it. She unzips her wet suit a bit and says, 'When did you last have a drink?' He said, 'Five years ago.' So she gets out a bottle of Scotch and he has a drink. Then she unzips her wet suit a bit more and says, 'And when was the last time you played around?' He looks at her in amazement and says: 'You're not telling me you've got a set of golf clubs in there?' Ronnie Corbett (1930 - 2016) |

#### pool_2

| object | 12-bit state | positive unit set | value | tied optima |
|---|---|---|---:|---:|
| target M_omega | `000000010010` | {u8, u11} | — | — |
| MCQ cap state | `100010000001` | {u1, u5, u12} | 0.023660 | 1 |
| behavioral cap state | `000000010011` | {u8, u11, u12} | 0.075291 | 4 |
| no-verbatim cap state | `110001110011` | {u1, u2, u6, u7, u8, u11, u12} | 0.072125 | 4 |
| best observed MCQ prompt | `101011011111` | {u1, u3, u5, u6, u8, u9, u10, u11, u12} | — | — |
| best observed behavioral prompt | `000000010011` | {u8, u11, u12} | — | — |

| unit | probe index | target | probe text |
|---|---:|---:|---|
| u1 | 1 | 0 | It's like the weather saw a state trooper It went from 90 to 45 real quick |
| u2 | 182 | 0 | Your mom is so fat Her school picture from first grade is still printing |
| u3 | 144 | 0 | What type of person names their child after a popular hotel chain? |
| u4 | 164 | 0 | What sort of mint do anarchists hate? Governmint |
| u5 | 199 | 0 | Did you know that after this next album, Matisyahu will be retiring? Soon he will be Jah-bless. |
| u6 | 99 | 0 | I finally realized I could no longer keep my broken money making machine. It just didn't make cents. |
| u7 | 232 | 0 | Importance of meditation When a wife keeps her head on a mans chest and slowly asks "Dear do have any other woman in your life" Remember, the answer is not important at this time Important is - the heartbeat, keep your heart beats in control Therefore, meditate regularly. |
| u8 | 294 | 1 | An Irishman walks into a bar and orders three shots of whiskey. The bartender asks him why he ordered three shots. "My life-long friends and I have a tradition. We grew up together but have since gone our separate ways. One is in England and one in the USA, but we each go into a bar on the same day every year and order three shots of whiskey. It's as if we are drinking them together." He then drinks the shots and leaves the bar. The next couple years, he returns and does the same. Then, one year the man returns but only orders two shots. He drinks them both. "I can't help but notice you only ordered two shots," the bartender said. "It appears you must have lost one of your friends. My condolences." "Oh no," the Irishman said. "Those chaps are doing fine. I just quit drinking, that's all." |
| u9 | 109 | 0 | A man had three horses One was black. One was white. One was eating. |
| u10 | 69 | 0 | Timing. Why can't drummers tell jokes? |
| u11 | 160 | 1 | I ate horse once by accident... My pulse started racing and my stomach was aching so I called my wife then went to the hospital. When she came to visit, she asked the Doctor how I was. He told her I was in “stable” condition. |
| u12 | 194 | 0 | If you set an episode of game of thrones away now, the opening credits will be coming to an end just when the clock strikes midnight on New Year's Eve. Start your year right. |

#### pool_3

| object | 12-bit state | positive unit set | value | tied optima |
|---|---|---|---:|---:|
| target M_omega | `100000010000` | {u1, u8} | — | — |
| MCQ cap state | `010001010110` | {u2, u6, u8, u10, u11} | 0.043456 | 1 |
| behavioral cap state | `001000000100` | {u3, u10} | 0.068517 | 8 |
| no-verbatim cap state | `000100000110` | {u4, u10, u11} | 0.048584 | 16 |
| best observed MCQ prompt | `010101000110` | {u2, u4, u6, u10, u11} | — | — |
| best observed behavioral prompt | `000000010100` | {u8, u10} | — | — |

| unit | probe index | target | probe text |
|---|---:|---:|---|
| u1 | 93 | 1 | What does a woman and kentucky fried chicken have in common?? By the time your finished with the breast and thighs, all you left is a greasy box to put your bone in. |
| u2 | 175 | 0 | Every time I walk into a store with my dad Worker: "Can I help you?" Dad: "No, he was born like that." |
| u3 | 55 | 0 | [Oh, yeah?] My ex-wife cheated on me with a communist! ...there were so many red flags. |
| u4 | 275 | 0 | What do you call a man with 6,022 x 10^23 dollars? A Moleionaire |
| u5 | 201 | 0 | What happens when you put two and two together? A Siamese orgy. |
| u6 | 219 | 0 | Someone asked me if I had ever noticed that I had a keen sense for being able to tell where water was underground... I replied, "I'm well aware." |
| u7 | 235 | 0 | Hay girl you know I am a horny praying mantis? |
| u8 | 40 | 1 | Out of all these emails I'm getting from businesses and companies about what they're doing to prevent the spread of coronavirus.... I'm mostly impressed to hear that our local massage parlor is beginning to use Purell as massage lubricant. I see they're still making money hand over fist. |
| u9 | 257 | 0 | Are you good at math..? ...then give me your number. |
| u10 | 255 | 0 | A renowned marine biologist is invited to a Great White conservation center The scientists there are very concerned that the Great Whites native to that area had all become sick and had developed digestive issues. The initial thinking was ocean pollution was to blame. The sharks weren't properly processing their meals and seemed to be evacuating their bowels at a much higher than normal frequency. Initially it seemed like it was only the infants that had been impacted but soon the adults seemed to be facing the issue as well. The Marine biologist is skeptical that pollution could cause such widespread issues so after some investigating he is able to figure out that there is actually a virus to blame that seems to be highly contagious and spreads through fecal matter. In other words, Baby shark do doodoo mommy shark do doodoo, daddy shark do doodoo... |
| u11 | 130 | 0 | A Buddhist goes to a hot dog vendor A Buddhist goes to a hot dog vendor, and the vendor asks him "Hey buddy what can I make ya?" "Make me one with everything" EDIT Punctuation |
| u12 | 48 | 0 | 2016 had many celebrity deaths.. ..but 2017 will Hurt the most. :( |

#### pool_4

| object | 12-bit state | positive unit set | value | tied optima |
|---|---|---|---:|---:|
| target M_omega | `000010000001` | {u5, u12} | — | — |
| MCQ cap state | `100011000100` | {u1, u5, u6, u10} | 0.003535 | 16 |
| behavioral cap state | `111010000000` | {u1, u2, u3, u5} | 0.049252 | 2 |
| no-verbatim cap state | `001001000000` | {u3, u6} | 0.067064 | 16 |
| best observed MCQ prompt | `100001111100` | {u1, u6, u7, u8, u9, u10} | — | — |
| best observed behavioral prompt | `000011000000` | {u5, u6} | — | — |

| unit | probe index | target | probe text |
|---|---:|---:|---|
| u1 | 288 | 0 | How did Columbus discover the New World? He sailed there occidentally. |
| u2 | 103 | 0 | I've determined that saying big words always will make you sound smart Totally photosynthesis right? |
| u3 | 123 | 0 | The other day this really fat guy was telling me he ran a marathon last week. But that just doesn't track. |
| u4 | 23 | 0 | What do you call it when a cow get's milked without consent? "Moo-lestation" |
| u5 | 126 | 1 | Fucking Doctors . . . A doctor walks into a nursing surgery. Say's to the mother, "I've got good news and i've got bad news." Mother says, "Give me the bad news." Doctor says, "Your baby was born with red hair" Mother says, "Give me the good news" Doctor says, "It died." |
| u6 | 179 | 0 | Knowledge is Power A jew once went on a walk with his son in a small town and said : -"My dear child, the rabbi told me "If you don't learn anything, you'll become nothing. But if you learn something you will be a well respected person, you will become a rabbi or get a high ranked job... but if you learn nothing, you will become a coachman, a servant or even a soldier."" At this moment a general passed by. The jew saw the general, pointed at him and said : -"You see, my child, thats how you will look!" (This joke was translated from Yiddish, I know it from jewish storyteller Fritz Muliar.) |
| u7 | 266 | 0 | For my 2017. New years resolution I'll go 3840 × 2160 |
| u8 | 100 | 0 | The soft spot on baby's head is basically an off button If you press hard enough |
| u9 | 155 | 0 | How do you cure a heart attack? You give your cardiac a rest. |
| u10 | 90 | 0 | What happened to the naughty wire? It was grounded... |
| u11 | 242 | 0 | I talked back to my OCD mother... she was very quick to put me in my place. |
| u12 | 148 | 1 | What does Gordon Ramsey shout when he sees a baby duckling gif on Reddit? IT'S FUCKING r/aww ! |

#### pool_5

| object | 12-bit state | positive unit set | value | tied optima |
|---|---|---|---:|---:|
| target M_omega | `000001001000` | {u6, u9} | — | — |
| MCQ cap state | `111000001111` | {u1, u2, u3, u9, u10, u11, u12} | 0.010314 | 2 |
| behavioral cap state | `011101011001` | {u2, u3, u4, u6, u8, u9, u12} | 0.043928 | 2 |
| no-verbatim cap state | `011100001011` | {u2, u3, u4, u9, u11, u12} | 0.049259 | 1 |
| best observed MCQ prompt | `101000001101` | {u1, u3, u9, u10, u12} | — | — |
| best observed behavioral prompt | `000001001001` | {u6, u9, u12} | — | — |

| unit | probe index | target | probe text |
|---|---:|---:|---|
| u1 | 174 | 0 | i poured root beer in a square glass... now i just have beer. |
| u2 | 176 | 0 | Bad news : I farted during my interview for bartender |
| u3 | 240 | 0 | Jello has created a product that deters insects. It's very effective, but the flavor is OFF-pudding. |
| u4 | 29 | 0 | I went to the Doctor and he said that I was paranoid He didn't actually say that but I knew what he was thinking. |
| u5 | 230 | 0 | A eunuch wants to commit suicide. He's told that he doesn't have the balls for this, so he gets prosthetic testicles. |
| u6 | 177 | 1 | How many chains does it take to enslave a black person in the twenty-first century? Two Chainz. |
| u7 | 264 | 0 | On her deathbed my wife said, "Sweety, I will see you in Heaven." Since then I have kicked a puppy, stole from 4 shops and set fire to an orphanage.! |
| u8 | 268 | 0 | Plane crash Two ladies, one white and one black is flying. Suddenly, the plane hits bad weather and the pilot declares that he is going to crash land. Hearing this, the black lady starts to undress. Surprised, the other ask, "what the hell are you doing? " She replies, "don't you know the first thing they look for after a crash is black box?". |
| u9 | 67 | 1 | Private investigator (At a fancy diner with wife and her friend) Wife's friend: So, what do you do for a living? Me: I'm a private investigator Wife: Kieth, you're allowed to say gynecologist Me: People are eating, Linda! |
| u10 | 36 | 0 | Boss: "and this is what you'll be making before taxes" Employee: "that's gross" |
| u11 | 84 | 0 | As a kid i looked up to Bill Nye the science guy, but as of today after learning about him i will probably look down on him. |
| u12 | 9 | 0 | What rock group has 4 men that do not sing? Mt. Rushmore |

#### pool_6

| object | 12-bit state | positive unit set | value | tied optima |
|---|---|---|---:|---:|
| target M_omega | `000000000101` | {u10, u12} | — | — |
| MCQ cap state | `010011011011` | {u2, u5, u6, u8, u9, u11, u12} | 0.010217 | 2 |
| behavioral cap state | `011010010110` | {u2, u3, u5, u8, u10, u11} | 0.054700 | 2 |
| no-verbatim cap state | `000111010001` | {u4, u5, u6, u8, u12} | 0.041062 | 2 |
| best observed MCQ prompt | `010010101111` | {u2, u5, u7, u9, u10, u11, u12} | — | — |
| best observed behavioral prompt | `100000100001` | {u1, u7, u12} | — | — |

| unit | probe index | target | probe text |
|---|---:|---:|---|
| u1 | 181 | 0 | Two police officers and their dog are walking down the street Two police officers and their dog are walking down the street. One of the officers turns to the other and asks, "Hey partner, how many penises does Buster have?" "Why, one of course" replies the other cop. "I'm not so sure about that, let's just check" says the first policeman. So they look down and sure enough they see only one penis. "That's strange" says the first cop, "this guy at the bar said 'Here goes that dog again with two dicks'" |
| u2 | 3 | 0 | Why was the snow yellow? Elsa let it go! |
| u3 | 284 | 0 | Kid : mom, how come you are white and I'm black ? Mom : if I can vaguely remember the things I did in that rave party, just be thankful that you are not barking!!! |
| u4 | 184 | 0 | a Country Boy visited the City and met a girl in a Bar who invited him back to her house, When they got there, she undressed and told him to get naked too. She said: "Let's start with a 69" The Country Boy replied: "What's that?" With that she got him into position, and they went at it Within a minute of starting, the City Girl felt a fart coming on She tried holding it back, but she figured the Country Boy was probably enjoying what she was doing to him and just let it rip Less than a Minute later, she felt another one coming on and since he hadn't said anything, let this one out as well After that, the Country Boy pushed her off, got up, and started getting dressed The City Girl, embarrassed, asked "I guess you didn't like that, huh?" The Country Boy said: "No, it was fine, but I just don't think I could take 67 more of those" |
| u5 | 57 | 0 | If Marty McFly had bipolar disorder... ....would that have made him Sweet n' Sour Chicken? |
| u6 | 166 | 0 | Whats the only recoil with the OK BOOMER? |
| u7 | 32 | 0 | How many people does it take to screw in a lightbulb? Two, but good luck finding a lightbulb that big. |
| u8 | 276 | 0 | What do whores and sailors have in common? They are both always surrounded by sea - men! I made up this one. |
| u9 | 269 | 0 | What did the motivational speaker ask the bottle of water? Do you have what it takes to be a liter? |
| u10 | 133 | 1 | I found my wife, my soulmate, my best friend on tinder I guess I wasn't invited to the orgy. |
| u11 | 171 | 0 | What type of yoga does Jesus do? Pontius Pilates. |
| u12 | 64 | 1 | A tourist in Hawaii is amazed at how healthy and invigorated he feels after just a few days into visiting the islands... He strikes up a conversation with one of the locals while they are wading out into the crystal clear, warm surf on yet another perfect island day. "I just cant get over how beautiful this place is," the tourist says excitedly, "I feel great! I haven't felt this young and healthy in years! Island life is fantastic!" The local says, "I know what you mean! Take me for instance. When I came here I was totally bald, didn't have any teeth and I couldn't even walk...and look at me now!" The tourist looks at him and says, "Wow, that's amazing! How long have you been here?" And the local says, "Oh, I was born here." |

### Metric 34: Use camera, editing, graphics, wardrobe, staging, music/bleeps/silence, and visual framing intentionally to set tone, misdirect, and time jokes.

#### pool_1

| object | 12-bit state | positive unit set | value | tied optima |
|---|---|---|---:|---:|
| target M_omega | `000000000100` | {u10} | — | — |
| MCQ cap state | `000000000000` | {empty} | 0.000000 | 4096 |
| behavioral cap state | `100000011010` | {u1, u8, u9, u11} | 0.013732 | 2 |
| no-verbatim cap state | `110010000100` | {u1, u2, u5, u10} | 0.012301 | 2 |
| best observed MCQ prompt | `000011101001` | {u5, u6, u7, u9, u12} | — | — |
| best observed behavioral prompt | `101111111011` | {u1, u3, u4, u5, u6, u7, u8, u9, u11, u12} | — | — |

| unit | probe index | target | probe text |
|---|---:|---:|---|
| u1 | 142 | 0 | The other night I was lying in bed, looking up at the stars, and I wondered, Where the hell is my roof? |
| u2 | 295 | 0 | Husband: Honey why are there broken condoms on the couch? |
| u3 | 192 | 0 | Why did no one laugh when queen farted at the dinner table? Because noble gasses cause no reaction |
| u4 | 60 | 0 | The artist, formerly known as E is now known as MC squared. |
| u5 | 223 | 0 | I have a friend who recently joined a religious yogurt group. He didn't realise it was Ya-Kult. |
| u6 | 81 | 0 | What do you call a family in which everyone from grand parents to grand kids smoke weed? Joint Family. |
| u7 | 246 | 0 | A college girl brings her new beau home to meet the family Her father takes him aside for a chat, "You seem like a nice enough fellow young man, what do you study?' "I'm a theology major sir." Answers the young man. "I see,If you dont mind my asking, where will you live with my little girl after you get married?" "God will provide." "What will you do for a living? How are you going to earn enough to put food on the table?" "God will provide" At the end of the evening, the girl goes up to her father and says " Daddy,I really like him,what do you think?" The father responds,"He seems like a nice enough fellow,but he seems to think I'm god." |
| u8 | 286 | 0 | Why did the lawyer go to culinary school? He wanted to be a sue chef |
| u9 | 14 | 0 | What is a squirrel's second favorite food? |
| u10 | 271 | 1 | A compulsive liar walks into a doctor's office claiming to be constipated... The doctor tells him he's full of shit. |
| u11 | 132 | 0 | "Oh, no", the husband replies. "She's left handed." A wife asks her husband, "Honey, if I died, would you remarry?" oh wait fuck |
| u12 | 149 | 0 | A man was marooned on a desert island. One day a beautiful woman arrives in a wet suit. 'When did you last have a smoke?' she asks. 'Five years ago.' So she gets out a cigar and he smokes it. She unzips her wet suit a bit and says, 'When did you last have a drink?' He said, 'Five years ago.' So she gets out a bottle of Scotch and he has a drink. Then she unzips her wet suit a bit more and says, 'And when was the last time you played around?' He looks at her in amazement and says: 'You're not telling me you've got a set of golf clubs in there?' Ronnie Corbett (1930 - 2016) |

#### pool_2

| object | 12-bit state | positive unit set | value | tied optima |
|---|---|---|---:|---:|
| target M_omega | `001000000000` | {u3} | — | — |
| MCQ cap state | `000000000000` | {empty} | 0.000000 | 4096 |
| behavioral cap state | `011001101000` | {u2, u3, u6, u7, u9} | 0.013943 | 1 |
| no-verbatim cap state | `010001011110` | {u2, u6, u8, u9, u10, u11} | 0.014899 | 1 |
| best observed MCQ prompt | `000000100000` | {u7} | — | — |
| best observed behavioral prompt | `100110101111` | {u1, u4, u5, u7, u9, u10, u11, u12} | — | — |

| unit | probe index | target | probe text |
|---|---:|---:|---|
| u1 | 1 | 0 | It's like the weather saw a state trooper It went from 90 to 45 real quick |
| u2 | 182 | 0 | Your mom is so fat Her school picture from first grade is still printing |
| u3 | 40 | 1 | Out of all these emails I'm getting from businesses and companies about what they're doing to prevent the spread of coronavirus.... I'm mostly impressed to hear that our local massage parlor is beginning to use Purell as massage lubricant. I see they're still making money hand over fist. |
| u4 | 199 | 0 | Did you know that after this next album, Matisyahu will be retiring? Soon he will be Jah-bless. |
| u5 | 99 | 0 | I finally realized I could no longer keep my broken money making machine. It just didn't make cents. |
| u6 | 232 | 0 | Importance of meditation When a wife keeps her head on a mans chest and slowly asks "Dear do have any other woman in your life" Remember, the answer is not important at this time Important is - the heartbeat, keep your heart beats in control Therefore, meditate regularly. |
| u7 | 294 | 0 | An Irishman walks into a bar and orders three shots of whiskey. The bartender asks him why he ordered three shots. "My life-long friends and I have a tradition. We grew up together but have since gone our separate ways. One is in England and one in the USA, but we each go into a bar on the same day every year and order three shots of whiskey. It's as if we are drinking them together." He then drinks the shots and leaves the bar. The next couple years, he returns and does the same. Then, one year the man returns but only orders two shots. He drinks them both. "I can't help but notice you only ordered two shots," the bartender said. "It appears you must have lost one of your friends. My condolences." "Oh no," the Irishman said. "Those chaps are doing fine. I just quit drinking, that's all." |
| u8 | 93 | 0 | What does a woman and kentucky fried chicken have in common?? By the time your finished with the breast and thighs, all you left is a greasy box to put your bone in. |
| u9 | 109 | 0 | A man had three horses One was black. One was white. One was eating. |
| u10 | 69 | 0 | Timing. Why can't drummers tell jokes? |
| u11 | 290 | 0 | I had a terrible nightmare last night that I ate a muffler. Today, I'm so exhausted. |
| u12 | 194 | 0 | If you set an episode of game of thrones away now, the opening credits will be coming to an end just when the clock strikes midnight on New Year's Eve. Start your year right. |

#### pool_3

| object | 12-bit state | positive unit set | value | tied optima |
|---|---|---|---:|---:|
| target M_omega | `000000000000` | {empty} | — | — |
| MCQ cap state | `000000000000` | {empty} | 0.000000 | 4096 |
| behavioral cap state | `101000011110` | {u1, u3, u8, u9, u10, u11} | 0.010359 | 4 |
| no-verbatim cap state | `101100000100` | {u1, u3, u4, u10} | 0.010684 | 4 |
| best observed MCQ prompt | `001000001000` | {u3, u9} | — | — |
| best observed behavioral prompt | `100010101110` | {u1, u5, u7, u9, u10, u11} | — | — |

| unit | probe index | target | probe text |
|---|---:|---:|---|
| u1 | 144 | 0 | What type of person names their child after a popular hotel chain? |
| u2 | 175 | 0 | Every time I walk into a store with my dad Worker: "Can I help you?" Dad: "No, he was born like that." |
| u3 | 164 | 0 | What sort of mint do anarchists hate? Governmint |
| u4 | 55 | 0 | [Oh, yeah?] My ex-wife cheated on me with a communist! ...there were so many red flags. |
| u5 | 275 | 0 | What do you call a man with 6,022 x 10^23 dollars? A Moleionaire |
| u6 | 201 | 0 | What happens when you put two and two together? A Siamese orgy. |
| u7 | 219 | 0 | Someone asked me if I had ever noticed that I had a keen sense for being able to tell where water was underground... I replied, "I'm well aware." |
| u8 | 235 | 0 | Hay girl you know I am a horny praying mantis? |
| u9 | 74 | 0 | A hole was discovered in the fence surrounding the local nudist colony Police are looking into it |
| u10 | 130 | 0 | A Buddhist goes to a hot dog vendor A Buddhist goes to a hot dog vendor, and the vendor asks him "Hey buddy what can I make ya?" "Make me one with everything" EDIT Punctuation |
| u11 | 48 | 0 | 2016 had many celebrity deaths.. ..but 2017 will Hurt the most. :( |
| u12 | 177 | 0 | How many chains does it take to enslave a black person in the twenty-first century? Two Chainz. |

#### pool_4

| object | 12-bit state | positive unit set | value | tied optima |
|---|---|---|---:|---:|
| target M_omega | `000000000000` | {empty} | — | — |
| MCQ cap state | `000000000000` | {empty} | 0.000000 | 4096 |
| behavioral cap state | `000000100011` | {u7, u11, u12} | 0.022025 | 2 |
| no-verbatim cap state | `110001100011` | {u1, u2, u6, u7, u11, u12} | 0.029850 | 2 |
| best observed MCQ prompt | `000000000001` | {u12} | — | — |
| best observed behavioral prompt | `000100110001` | {u4, u7, u8, u12} | — | — |

| unit | probe index | target | probe text |
|---|---:|---:|---|
| u1 | 288 | 0 | How did Columbus discover the New World? He sailed there occidentally. |
| u2 | 103 | 0 | I've determined that saying big words always will make you sound smart Totally photosynthesis right? |
| u3 | 123 | 0 | The other day this really fat guy was telling me he ran a marathon last week. But that just doesn't track. |
| u4 | 160 | 0 | I ate horse once by accident... My pulse started racing and my stomach was aching so I called my wife then went to the hospital. When she came to visit, she asked the Doctor how I was. He told her I was in “stable” condition. |
| u5 | 126 | 0 | Fucking Doctors . . . A doctor walks into a nursing surgery. Say's to the mother, "I've got good news and i've got bad news." Mother says, "Give me the bad news." Doctor says, "Your baby was born with red hair" Mother says, "Give me the good news" Doctor says, "It died." |
| u6 | 106 | 0 | Most girls are like modern computers They won’t accept my 3 1/2” floppy |
| u7 | 179 | 0 | Knowledge is Power A jew once went on a walk with his son in a small town and said : -"My dear child, the rabbi told me "If you don't learn anything, you'll become nothing. But if you learn something you will be a well respected person, you will become a rabbi or get a high ranked job... but if you learn nothing, you will become a coachman, a servant or even a soldier."" At this moment a general passed by. The jew saw the general, pointed at him and said : -"You see, my child, thats how you will look!" (This joke was translated from Yiddish, I know it from jewish storyteller Fritz Muliar.) |
| u8 | 100 | 0 | The soft spot on baby's head is basically an off button If you press hard enough |
| u9 | 257 | 0 | Are you good at math..? ...then give me your number. |
| u10 | 242 | 0 | I talked back to my OCD mother... she was very quick to put me in my place. |
| u11 | 148 | 0 | What does Gordon Ramsey shout when he sees a baby duckling gif on Reddit? IT'S FUCKING r/aww ! |
| u12 | 255 | 0 | A renowned marine biologist is invited to a Great White conservation center The scientists there are very concerned that the Great Whites native to that area had all become sick and had developed digestive issues. The initial thinking was ocean pollution was to blame. The sharks weren't properly processing their meals and seemed to be evacuating their bowels at a much higher than normal frequency. Initially it seemed like it was only the infants that had been impacted but soon the adults seemed to be facing the issue as well. The Marine biologist is skeptical that pollution could cause such widespread issues so after some investigating he is able to figure out that there is actually a virus to blame that seems to be highly contagious and spreads through fecal matter. In other words, Baby shark do doodoo mommy shark do doodoo, daddy shark do doodoo... |

#### pool_5

| object | 12-bit state | positive unit set | value | tied optima |
|---|---|---|---:|---:|
| target M_omega | `000000000000` | {empty} | — | — |
| MCQ cap state | `000000000000` | {empty} | 0.000000 | 4096 |
| behavioral cap state | `011111010000` | {u2, u3, u4, u5, u6, u8} | 0.023885 | 1 |
| no-verbatim cap state | `011101011110` | {u2, u3, u4, u6, u8, u9, u10, u11} | 0.023218 | 1 |
| best observed MCQ prompt | `100000001000` | {u1, u9} | — | — |
| best observed behavioral prompt | `111111001000` | {u1, u2, u3, u4, u5, u6, u9} | — | — |

| unit | probe index | target | probe text |
|---|---:|---:|---|
| u1 | 52 | 0 | A woman gets a call from kidnappers. "We have your son," said the kidnapper. "I don't have a son," says the woman. "Then who just asked for warm milk and made us cut the crust off his sandwiches?" "Oh, God you have my husband!" |
| u2 | 174 | 0 | i poured root beer in a square glass... now i just have beer. |
| u3 | 266 | 0 | For my 2017. New years resolution I'll go 3840 × 2160 |
| u4 | 90 | 0 | What happened to the naughty wire? It was grounded... |
| u5 | 176 | 0 | Bad news : I farted during my interview for bartender |
| u6 | 102 | 0 | A Racist Joke Two racists are sitting on a park bench, watching and making commentary on passersby. "You know," the first racist says, "we're the real victims in today's society." The second racist nods knowingly. "As soon as someone hears that you have a problem with one group of people or another," the first racist continues, "they make all kinds of unfair and unfounded assumptions about you." Once again, the second racist nods. "Just as an example," says the first racist, "if someone were to read a transcript of our conversation, they'd almost certainly think that both of us were white." The second racist grins and says "哈哈！我是公交车司机!" |
| u7 | 29 | 0 | I went to the Doctor and he said that I was paranoid He didn't actually say that but I knew what he was thinking. |
| u8 | 230 | 0 | A eunuch wants to commit suicide. He's told that he doesn't have the balls for this, so he gets prosthetic testicles. |
| u9 | 155 | 0 | How do you cure a heart attack? You give your cardiac a rest. |
| u10 | 264 | 0 | On her deathbed my wife said, "Sweety, I will see you in Heaven." Since then I have kicked a puppy, stole from 4 shops and set fire to an orphanage.! |
| u11 | 268 | 0 | Plane crash Two ladies, one white and one black is flying. Suddenly, the plane hits bad weather and the pilot declares that he is going to crash land. Hearing this, the black lady starts to undress. Surprised, the other ask, "what the hell are you doing? " She replies, "don't you know the first thing they look for after a crash is black box?". |
| u12 | 23 | 0 | What do you call it when a cow get's milked without consent? "Moo-lestation" |

#### pool_6

| object | 12-bit state | positive unit set | value | tied optima |
|---|---|---|---:|---:|
| target M_omega | `000000000000` | {empty} | — | — |
| MCQ cap state | `000000000000` | {empty} | 0.000000 | 4096 |
| behavioral cap state | `111101111010` | {u1, u2, u3, u4, u6, u7, u8, u9, u11} | 0.011172 | 1 |
| no-verbatim cap state | `001101000010` | {u3, u4, u6, u11} | 0.017036 | 1 |
| best observed MCQ prompt | `010010000001` | {u2, u5, u12} | — | — |
| best observed behavioral prompt | `110111001111` | {u1, u2, u4, u5, u6, u9, u10, u11, u12} | — | — |

| unit | probe index | target | probe text |
|---|---:|---:|---|
| u1 | 181 | 0 | Two police officers and their dog are walking down the street Two police officers and their dog are walking down the street. One of the officers turns to the other and asks, "Hey partner, how many penises does Buster have?" "Why, one of course" replies the other cop. "I'm not so sure about that, let's just check" says the first policeman. So they look down and sure enough they see only one penis. "That's strange" says the first cop, "this guy at the bar said 'Here goes that dog again with two dicks'" |
| u2 | 3 | 0 | Why was the snow yellow? Elsa let it go! |
| u3 | 184 | 0 | a Country Boy visited the City and met a girl in a Bar who invited him back to her house, When they got there, she undressed and told him to get naked too. She said: "Let's start with a 69" The Country Boy replied: "What's that?" With that she got him into position, and they went at it Within a minute of starting, the City Girl felt a fart coming on She tried holding it back, but she figured the Country Boy was probably enjoying what she was doing to him and just let it rip Less than a Minute later, she felt another one coming on and since he hadn't said anything, let this one out as well After that, the Country Boy pushed her off, got up, and started getting dressed The City Girl, embarrassed, asked "I guess you didn't like that, huh?" The Country Boy said: "No, it was fine, but I just don't think I could take 67 more of those" |
| u4 | 240 | 0 | Jello has created a product that deters insects. It's very effective, but the flavor is OFF-pudding. |
| u5 | 9 | 0 | What rock group has 4 men that do not sing? Mt. Rushmore |
| u6 | 67 | 0 | Private investigator (At a fancy diner with wife and her friend) Wife's friend: So, what do you do for a living? Me: I'm a private investigator Wife: Kieth, you're allowed to say gynecologist Me: People are eating, Linda! |
| u7 | 166 | 0 | Whats the only recoil with the OK BOOMER? |
| u8 | 137 | 0 | Three guys die... and Saint Peter stops them at the Golden Gates. He tells them, "Depending how faithful you were to your wife, depends what kind of car you drive across the Golden Bridge to heaven." &amp;#x200B; First guy says, "I was married 10 years and only cheated three times." &amp;#x200B; Saint Peter says, "That's ok I suppose, here take this older model pick-up truck." &amp;#x200B; Second guy says, "I was married 15 years and only cheated once!" &amp;#x200B; Saint Peter says, "Pretty great, here take this sports car." &amp;#x200B; Third guy says, "I was married 40 years and never cheated on my wife." &amp;#x200B; Saint Peter says, "Wow that's the best I've ever heard! Here, take this Golden Edition Rolls-Royce." &amp;#x200B; The three guys start across the bridge and the Rolls takes off and leaves them. About half way across, the other two guys find the Rolls pulled over with his head on the steering wheel. They stop and walk over. &amp;#x200B; First guy says, "Come on man, being dead isn't so bad." &amp;#x200B; Second guy says, "Yeah, look what you're driving, and look what we're driving." &amp;#x200B; Third guy says, "No guys, you don't get it, I just saw my wife go by on a skateboard!" |
| u9 | 36 | 0 | Boss: "and this is what you'll be making before taxes" Employee: "that's gross" |
| u10 | 248 | 0 | Robert was arrested for stealing an 100 inch TV and a dishwasher from his fiance While in the police car, a policeman turned to him and said "What were you thinking? Were you drunk?" He immediately replied: "No, she asked me to do it!" "How?!", answered the policeman. "We were having a conversation and she said 'Oh please Rob!'" |
| u11 | 84 | 0 | As a kid i looked up to Bill Nye the science guy, but as of today after learning about him i will probably look down on him. |
| u12 | 269 | 0 | What did the motivational speaker ask the bottle of water? Do you have what it takes to be a liter? |

### Metric 50: Prime a fair initial reading and conceal an alternate interpretation, then flip expectations with a justified, non‑obvious reveal; misdirection, timing, and clarity land cleanly.

#### pool_1

| object | 12-bit state | positive unit set | value | tied optima |
|---|---|---|---:|---:|
| target M_omega | `001000000101` | {u3, u10, u12} | — | — |
| MCQ cap state | `000000000000` | {empty} | 0.000000 | 4096 |
| behavioral cap state | `100110000111` | {u1, u4, u5, u10, u11, u12} | 0.041087 | 16 |
| no-verbatim cap state | `010100100100` | {u2, u4, u7, u10} | 0.078412 | 8 |
| best observed MCQ prompt | `001001001001` | {u3, u6, u9, u12} | — | — |
| best observed behavioral prompt | `000001100001` | {u6, u7, u12} | — | — |

| unit | probe index | target | probe text |
|---|---:|---:|---|
| u1 | 142 | 0 | The other night I was lying in bed, looking up at the stars, and I wondered, Where the hell is my roof? |
| u2 | 295 | 0 | Husband: Honey why are there broken condoms on the couch? |
| u3 | 52 | 1 | A woman gets a call from kidnappers. "We have your son," said the kidnapper. "I don't have a son," says the woman. "Then who just asked for warm milk and made us cut the crust off his sandwiches?" "Oh, God you have my husband!" |
| u4 | 290 | 0 | I had a terrible nightmare last night that I ate a muffler. Today, I'm so exhausted. |
| u5 | 60 | 0 | The artist, formerly known as E is now known as MC squared. |
| u6 | 223 | 0 | I have a friend who recently joined a religious yogurt group. He didn't realise it was Ya-Kult. |
| u7 | 81 | 0 | What do you call a family in which everyone from grand parents to grand kids smoke weed? Joint Family. |
| u8 | 286 | 0 | Why did the lawyer go to culinary school? He wanted to be a sue chef |
| u9 | 14 | 0 | What is a squirrel's second favorite food? |
| u10 | 271 | 1 | A compulsive liar walks into a doctor's office claiming to be constipated... The doctor tells him he's full of shit. |
| u11 | 132 | 0 | "Oh, no", the husband replies. "She's left handed." A wife asks her husband, "Honey, if I died, would you remarry?" oh wait fuck |
| u12 | 149 | 1 | A man was marooned on a desert island. One day a beautiful woman arrives in a wet suit. 'When did you last have a smoke?' she asks. 'Five years ago.' So she gets out a cigar and he smokes it. She unzips her wet suit a bit and says, 'When did you last have a drink?' He said, 'Five years ago.' So she gets out a bottle of Scotch and he has a drink. Then she unzips her wet suit a bit more and says, 'And when was the last time you played around?' He looks at her in amazement and says: 'You're not telling me you've got a set of golf clubs in there?' Ronnie Corbett (1930 - 2016) |

#### pool_2

| object | 12-bit state | positive unit set | value | tied optima |
|---|---|---|---:|---:|
| target M_omega | `000001000110` | {u6, u10, u11} | — | — |
| MCQ cap state | `000001000001` | {u6, u12} | 0.000620 | 16 |
| behavioral cap state | `000100010010` | {u4, u8, u11} | 0.078224 | 4 |
| no-verbatim cap state | `000111000100` | {u4, u5, u6, u10} | 0.103547 | 4 |
| best observed MCQ prompt | `000001000001` | {u6, u12} | — | — |
| best observed behavioral prompt | `101111110111` | {u1, u3, u4, u5, u6, u7, u8, u10, u11, u12} | — | — |

| unit | probe index | target | probe text |
|---|---:|---:|---|
| u1 | 1 | 0 | It's like the weather saw a state trooper It went from 90 to 45 real quick |
| u2 | 182 | 0 | Your mom is so fat Her school picture from first grade is still printing |
| u3 | 164 | 0 | What sort of mint do anarchists hate? Governmint |
| u4 | 199 | 0 | Did you know that after this next album, Matisyahu will be retiring? Soon he will be Jah-bless. |
| u5 | 99 | 0 | I finally realized I could no longer keep my broken money making machine. It just didn't make cents. |
| u6 | 294 | 1 | An Irishman walks into a bar and orders three shots of whiskey. The bartender asks him why he ordered three shots. "My life-long friends and I have a tradition. We grew up together but have since gone our separate ways. One is in England and one in the USA, but we each go into a bar on the same day every year and order three shots of whiskey. It's as if we are drinking them together." He then drinks the shots and leaves the bar. The next couple years, he returns and does the same. Then, one year the man returns but only orders two shots. He drinks them both. "I can't help but notice you only ordered two shots," the bartender said. "It appears you must have lost one of your friends. My condolences." "Oh no," the Irishman said. "Those chaps are doing fine. I just quit drinking, that's all." |
| u7 | 93 | 0 | What does a woman and kentucky fried chicken have in common?? By the time your finished with the breast and thighs, all you left is a greasy box to put your bone in. |
| u8 | 74 | 0 | A hole was discovered in the fence surrounding the local nudist colony Police are looking into it |
| u9 | 69 | 0 | Timing. Why can't drummers tell jokes? |
| u10 | 177 | 1 | How many chains does it take to enslave a black person in the twenty-first century? Two Chainz. |
| u11 | 160 | 1 | I ate horse once by accident... My pulse started racing and my stomach was aching so I called my wife then went to the hospital. When she came to visit, she asked the Doctor how I was. He told her I was in “stable” condition. |
| u12 | 194 | 0 | If you set an episode of game of thrones away now, the opening credits will be coming to an end just when the clock strikes midnight on New Year's Eve. Start your year right. |

#### pool_3

| object | 12-bit state | positive unit set | value | tied optima |
|---|---|---|---:|---:|
| target M_omega | `001000010100` | {u3, u8, u10} | — | — |
| MCQ cap state | `000000000000` | {empty} | 0.000000 | 4096 |
| behavioral cap state | `001100101100` | {u3, u4, u7, u9, u10} | 0.048251 | 4 |
| no-verbatim cap state | `000000111100` | {u7, u8, u9, u10} | 0.050443 | 64 |
| best observed MCQ prompt | `000000010000` | {u8} | — | — |
| best observed behavioral prompt | `001110001100` | {u3, u4, u5, u9, u10} | — | — |

| unit | probe index | target | probe text |
|---|---:|---:|---|
| u1 | 144 | 0 | What type of person names their child after a popular hotel chain? |
| u2 | 175 | 0 | Every time I walk into a store with my dad Worker: "Can I help you?" Dad: "No, he was born like that." |
| u3 | 133 | 1 | I found my wife, my soulmate, my best friend on tinder I guess I wasn't invited to the orgy. |
| u4 | 55 | 0 | [Oh, yeah?] My ex-wife cheated on me with a communist! ...there were so many red flags. |
| u5 | 275 | 0 | What do you call a man with 6,022 x 10^23 dollars? A Moleionaire |
| u6 | 201 | 0 | What happens when you put two and two together? A Siamese orgy. |
| u7 | 219 | 0 | Someone asked me if I had ever noticed that I had a keen sense for being able to tell where water was underground... I replied, "I'm well aware." |
| u8 | 109 | 1 | A man had three horses One was black. One was white. One was eating. |
| u9 | 40 | 0 | Out of all these emails I'm getting from businesses and companies about what they're doing to prevent the spread of coronavirus.... I'm mostly impressed to hear that our local massage parlor is beginning to use Purell as massage lubricant. I see they're still making money hand over fist. |
| u10 | 67 | 1 | Private investigator (At a fancy diner with wife and her friend) Wife's friend: So, what do you do for a living? Me: I'm a private investigator Wife: Kieth, you're allowed to say gynecologist Me: People are eating, Linda! |
| u11 | 130 | 0 | A Buddhist goes to a hot dog vendor A Buddhist goes to a hot dog vendor, and the vendor asks him "Hey buddy what can I make ya?" "Make me one with everything" EDIT Punctuation |
| u12 | 48 | 0 | 2016 had many celebrity deaths.. ..but 2017 will Hurt the most. :( |

#### pool_4

| object | 12-bit state | positive unit set | value | tied optima |
|---|---|---|---:|---:|
| target M_omega | `100001000100` | {u1, u6, u10} | — | — |
| MCQ cap state | `000000000000` | {empty} | 0.000000 | 4096 |
| behavioral cap state | `011001001100` | {u2, u3, u6, u9, u10} | 0.081177 | 4 |
| no-verbatim cap state | `000101100110` | {u4, u6, u7, u10, u11} | 0.096081 | 2 |
| best observed MCQ prompt | `000000000000` | {empty} | — | — |
| best observed behavioral prompt | `010001000101` | {u2, u6, u10, u12} | — | — |

| unit | probe index | target | probe text |
|---|---:|---:|---|
| u1 | 32 | 1 | How many people does it take to screw in a lightbulb? Two, but good luck finding a lightbulb that big. |
| u2 | 288 | 0 | How did Columbus discover the New World? He sailed there occidentally. |
| u3 | 123 | 0 | The other day this really fat guy was telling me he ran a marathon last week. But that just doesn't track. |
| u4 | 235 | 0 | Hay girl you know I am a horny praying mantis? |
| u5 | 126 | 0 | Fucking Doctors . . . A doctor walks into a nursing surgery. Say's to the mother, "I've got good news and i've got bad news." Mother says, "Give me the bad news." Doctor says, "Your baby was born with red hair" Mother says, "Give me the good news" Doctor says, "It died." |
| u6 | 248 | 1 | Robert was arrested for stealing an 100 inch TV and a dishwasher from his fiance While in the police car, a policeman turned to him and said "What were you thinking? Were you drunk?" He immediately replied: "No, she asked me to do it!" "How?!", answered the policeman. "We were having a conversation and she said 'Oh please Rob!'" |
| u7 | 106 | 0 | Most girls are like modern computers They won’t accept my 3 1/2” floppy |
| u8 | 100 | 0 | The soft spot on baby's head is basically an off button If you press hard enough |
| u9 | 257 | 0 | Are you good at math..? ...then give me your number. |
| u10 | 64 | 1 | A tourist in Hawaii is amazed at how healthy and invigorated he feels after just a few days into visiting the islands... He strikes up a conversation with one of the locals while they are wading out into the crystal clear, warm surf on yet another perfect island day. "I just cant get over how beautiful this place is," the tourist says excitedly, "I feel great! I haven't felt this young and healthy in years! Island life is fantastic!" The local says, "I know what you mean! Take me for instance. When I came here I was totally bald, didn't have any teeth and I couldn't even walk...and look at me now!" The tourist looks at him and says, "Wow, that's amazing! How long have you been here?" And the local says, "Oh, I was born here." |
| u11 | 242 | 0 | I talked back to my OCD mother... she was very quick to put me in my place. |
| u12 | 148 | 0 | What does Gordon Ramsey shout when he sees a baby duckling gif on Reddit? IT'S FUCKING r/aww ! |

#### pool_5

| object | 12-bit state | positive unit set | value | tied optima |
|---|---|---|---:|---:|
| target M_omega | `001000000100` | {u3, u10} | — | — |
| MCQ cap state | `000000000000` | {empty} | 0.000000 | 4096 |
| behavioral cap state | `010000101001` | {u2, u7, u9, u12} | 0.075958 | 4 |
| no-verbatim cap state | `001010001011` | {u3, u5, u9, u11, u12} | 0.091314 | 4 |
| best observed MCQ prompt | `000000000000` | {empty} | — | — |
| best observed behavioral prompt | `001101000010` | {u3, u4, u6, u11} | — | — |

| unit | probe index | target | probe text |
|---|---:|---:|---|
| u1 | 266 | 0 | For my 2017. New years resolution I'll go 3840 × 2160 |
| u2 | 90 | 0 | What happened to the naughty wire? It was grounded... |
| u3 | 58 | 1 | Bag Lady A little old lady was walking down the street dragging two large plastic garbage bags behind her. One of the bags was ripped and every once in awhile a $20 bill fell out onto the sidewalk. A policeman stopped her, and said, "Ma'am, there are $20 bills falling out of that bag." "Oh, really? Darn it!" said the little old lady. "I'd better go back and pick them up. Thanks for telling me, Officer." "Not so fast," said the cop. "Where did you get all that money? Did you steal it?" "Oh, no, no", said the old lady. "You see, my back yard is right next to a golf course. A lot of golfers come and pee through a knot hole in my fence, right into my flower garden. It kills the flowers, you know. Then I thought, 'why not make the best of it?' So, now, I stand behind the fence by the knot hole, real quiet, with my hedge clippers. Every time some guy sticks his thing through my fence, I surprise him, grab hold of it and say, 'O.K., buddy! Give me $20 or off it comes!' "Well, that seems only fair," said the cop, laughing. "OK. Good luck! Oh, by the way, what's in the other bag?" "Not everybody pays." |
| u4 | 103 | 0 | I've determined that saying big words always will make you sound smart Totally photosynthesis right? |
| u5 | 176 | 0 | Bad news : I farted during my interview for bartender |
| u6 | 240 | 0 | Jello has created a product that deters insects. It's very effective, but the flavor is OFF-pudding. |
| u7 | 29 | 0 | I went to the Doctor and he said that I was paranoid He didn't actually say that but I knew what he was thinking. |
| u8 | 230 | 0 | A eunuch wants to commit suicide. He's told that he doesn't have the balls for this, so he gets prosthetic testicles. |
| u9 | 155 | 0 | How do you cure a heart attack? You give your cardiac a rest. |
| u10 | 268 | 1 | Plane crash Two ladies, one white and one black is flying. Suddenly, the plane hits bad weather and the pilot declares that he is going to crash land. Hearing this, the black lady starts to undress. Surprised, the other ask, "what the hell are you doing? " She replies, "don't you know the first thing they look for after a crash is black box?". |
| u11 | 23 | 0 | What do you call it when a cow get's milked without consent? "Moo-lestation" |
| u12 | 84 | 0 | As a kid i looked up to Bill Nye the science guy, but as of today after learning about him i will probably look down on him. |

#### pool_6

| object | 12-bit state | positive unit set | value | tied optima |
|---|---|---|---:|---:|
| target M_omega | `100000100000` | {u1, u7} | — | — |
| MCQ cap state | `000000000000` | {empty} | 0.000000 | 4096 |
| behavioral cap state | `000010101011` | {u5, u7, u9, u11, u12} | 0.029608 | 24 |
| no-verbatim cap state | `111010001001` | {u1, u2, u3, u5, u9, u12} | 0.028909 | 8 |
| best observed MCQ prompt | `100000100000` | {u1, u7} | — | — |
| best observed behavioral prompt | `111000110001` | {u1, u2, u3, u7, u8, u12} | — | — |

| unit | probe index | target | probe text |
|---|---:|---:|---|
| u1 | 174 | 1 | i poured root beer in a square glass... now i just have beer. |
| u2 | 3 | 0 | Why was the snow yellow? Elsa let it go! |
| u3 | 284 | 0 | Kid : mom, how come you are white and I'm black ? Mom : if I can vaguely remember the things I did in that rave party, just be thankful that you are not barking!!! |
| u4 | 57 | 0 | If Marty McFly had bipolar disorder... ....would that have made him Sweet n' Sour Chicken? |
| u5 | 24 | 0 | Found the moron that doesn't know what "thou" means. It's obviously you. |
| u6 | 72 | 0 | Everyone tells me I'm average... That's just mean. |
| u7 | 9 | 1 | What rock group has 4 men that do not sing? Mt. Rushmore |
| u8 | 166 | 0 | Whats the only recoil with the OK BOOMER? |
| u9 | 63 | 0 | Boy catches a priest masturbating and asks, "What are you doing father?" |
| u10 | 276 | 0 | What do whores and sailors have in common? They are both always surrounded by sea - men! I made up this one. |
| u11 | 36 | 0 | Boss: "and this is what you'll be making before taxes" Employee: "that's gross" |
| u12 | 171 | 0 | What type of yoga does Jesus do? Pontius Pilates. |

## What was and was not controlled

- Included: blind/no-demo subtraction, deterministic shuffled-label subtraction, eight position-counterbalanced MCQ option orders, unconstrained versus no-verbatim induction arms, disjoint teaching pools and held-out H, candidate-prompt non-disclosure, exhaustive finite state tables, and exact pool/panel sensitivity.
- Diagnostics only: blind-prior balance, best-explanation rate, constant-prediction degeneracy, and pool variance.
- Not available in this run: gold-fidelity labels.
- Not run as part of v13.1: the older paired paraphrase/persona/format form-invariance census. The no-verbatim arm and option-order block are narrower controls and should not be described as full form invariance.
