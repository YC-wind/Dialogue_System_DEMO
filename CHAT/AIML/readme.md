# AIML

AIML stands for Artificial Intelligence Markup Language. AIML was developed by the Alicebot free software community and Dr. Richard S. Wallace during 1995-2000. AIML is used to create or customize Alicebot which is a chat-box application based on A.L.I.C.E. (Artificial Linguistic Internet Computer Entity) free software.

AIML代表人工智能标记语言。AIML由Alicebot自由软件社区和Richard S. Wallace博士在1995-2000期间开发。AIML用于创建或定制Alicebot，这是一个基于ALICE（人工语言互联网计算机实体）免费软件的聊天框应用程序。

# introduction

Following are the important tags which are commonly used in AIML documents.

| S.No |AIML Tag|Description|
|:----:|:----:|:-----:|
|  1   |\<aiml> |Defines the beginning and end of a AIML document.|
|  2   |\<category>|Defines the unit of knowledge in Alicebot's knowledge base.|
|  3   |\<pattern>| efines the pattern to match what a user may input to an Alicebot.|
|4|\<template>|Defines the response of an Alicebot to user's input.|

Following are some of the other widely used aiml tags. We'll be discussing each tag in details in coming chapters.

| S.No |AIML Tag|Description|
|:----:|:----:|:-----:|
|  1   |\<star> |Used to match wild card * character(s) in the <pattern> Tag.|
|  2   |\<srai>|Multipurpose tag, used to call/match the other categories.|
|  3   |\<random>|Used <random> to get random responses.|
|  4   |\<li>|Used to represent multiple responses.|
|  5   |\<set> |Used to set value in an AIML variable.|
|  6   |\<get>|Used to get value stored in an AIML variable.|
|  7   |\<that>| Used in AIML to respond based on the context.|
|  8   |\<topic>|Used in AIML to store a context so that later conversation can be done based on that context.|
|  9   |\<think> |Used in AIML to store a variable without notifying the user.|
|  10  |\<condition>|Similar to switch statements in programming language. It helps ALICE to respond to matching input.|

## some example

```xml
<category>
   <pattern> HELLO ALICE </pattern>
   <template>
      Hello User
   </template>  
</category>
```
If the user enters "HELLO ALICE" then bot will respond as "Hello User"


```xml
<category>
   <pattern> A * is a *. </pattern>
   <template>
      When a <star index = "1"/> is not a <star index = "2"/>?
   </template>
</category>
```
If the user enters "A mango is a fruit." then bot will respond as "When a mango is not a fruit?"

```xml
<category>
      <pattern>HI</pattern>
      <template>
         <random>
            <li> Hello! </li>
            <li> Hi! Nice to meet you! </li>
         </random>
      </template>
 </category>
```
If the user enters "HI" then bot will random respond as "Hello!" or "Hi! Nice to meet you!"

```xml
<category>
      <pattern>I am *</pattern>
      <template>
         Hello <set name = "username"> <star/>! </set>
      </template>  
   </category>  
<category>
      <pattern>Good Night</pattern>
      <template>
         Hi <get name = "username"/> Thanks for the conversation!
      </template>  
</category>  
```
Human: I am Mahesh
Robot: Hello Mahesh!
Human: Good Night
Robot: Good Night Mahesh! Thanks for the conversation!

```xml
<?xml version = "1.0" encoding = "UTF-8"?>
<aiml version = "1.0.1" encoding = "UTF-8"?>
   <category>
      <pattern>WHAT ABOUT MOVIES</pattern>
      <template>Do you like comedy movies</template>  
   </category>
   
   <category>
      <pattern>YES</pattern>
      <that>Do you like comedy movies</that>
      <template>Nice, I like comedy movies too.</template>
   </category>
   
   <category>
      <pattern>NO</pattern>
      <that>Do you like comedy movies</that>
      <template>Ok! But I like comedy movies.</template>
   </category> 
   
</aiml>
```
Human: What about movies?
Robot: Do you like comedy movies?
Human: No
Robot: Ok! But I like comedy movies.

```xml
<?xml version = "1.0" encoding = "UTF-8"?>
<aiml version = "1.0.1" encoding = "UTF-8"?>
   <category>
      <pattern>LET DISCUSS MOVIES</pattern>
      <template>Yes <set name = "topic">movies</set></template>  
   </category>
   
   <topic name = "movies">
      <category>
         <pattern> * </pattern>
         <template>Watching good movie refreshes our minds.</template>
      </category>
      
      <category>
         <pattern> I LIKE WATCHING COMEDY! </pattern>
         <template>I like comedy movies too.</template>
      </category>
      
   </topic>
</aiml>
```
Human: let discuss movies
Robot: Yes movies
Human: Comedy movies are nice to watch
Robot: Watching good movie refreshes our minds.
Human: I like watching comedy
Robot: I too like watching comedy.


```xml
<?xml version = "1.0" encoding = "UTF-8"?>
<aiml version = "1.0.1" encoding = "UTF-8"?>
   <category>
      <pattern>My name is *</pattern>
      <template>
         Hello!<think><set name = "username"> <star/></set></think>
      </template>  
   </category>  
   
   <category>
      <pattern>Byeee</pattern>
      <template>
         Hi <get name = "username"/> Thanks for the conversation!
      </template>  
   </category>  
   
</aiml>
```
Human: My name is Mahesh
Robot: Hello!
Human: Byeee
Robot: Hi Mahesh Thanks for the conversation!


```xml
<?xml version = "1.0" encoding = "UTF-8"?>
<aiml version = "1.0.1" encoding = "UTF-8"?>
   <category>
      <pattern> HOW ARE YOU FEELING TODAY </pattern>
      
      <template>
         <think><set name = "state"> happy</set></think>
         <condition name = "state" value = "happy">
            I am happy!
         </condition>
         
         <condition name = "state" value = "sad">
            I am sad!
         </condition>
      </template>
      
   </category>
</aiml>
```
Human: How are you feeling today
Robot: I am happy!



# reference

[aiml教程](https://www.tutorialspoint.com/aiml/)