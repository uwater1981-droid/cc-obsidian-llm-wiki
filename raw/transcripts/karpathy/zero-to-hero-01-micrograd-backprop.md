---
title: "The spelled-out intro to neural networks and backpropagation: building micrograd"
video_id: VMj-3S1tku0
url: "https://www.youtube.com/watch?v=VMj-3S1tku0"
author: Andrej Karpathy
slug: zero-to-hero-01-micrograd-backprop
fetched_at: "2026-04-17T20:58:26+08:00"
type: youtube-transcript-whisper
transcript_source: "https://github.com/averkij/karcaps (001-large.html)"
segments: 1335
---

# The spelled-out intro to neural networks and backpropagation: building micrograd

> Video: https://www.youtube.com/watch?v=VMj-3S1tku0
> Transcript: averkij/karcaps (Whisper-large) `001-large.html`

[00:00:00.000] Hello, my name is Andre and I've been training deep neural networks for a bit more than a decade

[00:00:04.800] and in this lecture I'd like to show you what neural network training looks like under the

[00:00:08.880] hood. So in particular we are going to start with a blank Jupyter notebook and by the end of this

[00:00:13.920] lecture we will define and train a neural net and you'll get to see everything that goes on under

[00:00:18.720] the hood and exactly sort of how that works on an intuitive level. Now specifically what I would

[00:00:23.600] like to do is I would like to take you through building of micrograd. Now micrograd is this

[00:00:29.440] library that I released on github about two years ago but at the time I only uploaded the source

[00:00:34.160] code and you'd have to go and buy yourself and really figure out how it works. So in this lecture

[00:00:40.160] I will take you through it step by step and kind of comment on all the pieces of it. So what's

[00:00:44.560] micrograd and why is it interesting? Micrograd is basically an autograd engine. Autograd is short

[00:00:52.800] for automatic gradient and really what it does is it implements backpropagation. Now backpropagation

[00:00:58.320] is this algorithm that allows you to efficiently evaluate the gradient of some kind of a loss

[00:01:04.320] function with respect to the weights of a neural network and what that allows us to do then is we

[00:01:09.840] can iteratively tune the weights of that neural network to minimize the loss function and therefore

[00:01:14.320] improve the accuracy of the network. So backpropagation would be at the mathematical

[00:01:18.960] core of any modern deep neural network library like say PyTorch or JAX. So the functionality

[00:01:24.800] of micrograd is I think best illustrated by an example. So if we just scroll down here

[00:01:29.760] you'll see that micrograd basically allows you to build out mathematical expressions

[00:01:34.240] and here what we are doing is we have an expression that we're building out where you

[00:01:38.080] have two inputs a and b and you'll see that a and b are negative 4 and 2 but we are wrapping those

[00:01:46.160] values into this value object that we are going to build out as part of micrograd.

[00:01:50.480] So this value object will wrap the numbers themselves and then we are going to build

[00:01:55.520] out a mathematical expression here where a and b are transformed into c d and eventually e f and g

[00:02:03.360] and I'm showing some of the functionality of micrograd and the operations that it supports.

[00:02:08.160] So you can add two value objects, you can multiply them, you can raise them to a constant power,

[00:02:13.920] you can offset by one, negate, squash at zero, square, divide by constant, divide by it, etc.

[00:02:22.240] And so we're building out an expression graph with these two inputs a and b and we're creating

[00:02:27.600] an output value of g and micrograd will in the background build out this entire mathematical

[00:02:33.600] expression. So it will for example know that c is also a value, c was a result of an addition

[00:02:39.680] operation and the child nodes of c are a and b because it will maintain pointers to a and b

[00:02:48.160] value objects. So we'll basically know exactly how all of this is laid out and then not only can we

[00:02:54.080] do what we call the forward pass where we actually look at the value of g of course, that's pretty

[00:02:58.160] straightforward, we will access that using the dot data attribute and so the output of the forward

[00:03:04.080] pass, the value of g, is 24.7 it turns out, but the big deal is that we can also take this g value

[00:03:11.280] object and we can call dot backward and this will basically initialize backpropagation at the node g.

[00:03:18.960] And what backpropagation is going to do is it's going to start at g and it's going to go

[00:03:23.040] backwards through that expression graph and it's going to recursively apply the chain rule from

[00:03:27.840] calculus and what that allows us to do then is we're going to evaluate basically the derivative

[00:03:34.000] of g with respect to all the internal nodes like e d and c but also with respect to the inputs a

[00:03:41.120] and b and then we can actually query this derivative of g with respect to a for example

[00:03:47.360] that's a dot grad in this case it happens to be 138 and the derivative of g with respect to b

[00:03:53.040] which also happens to be here 645 and this derivative we'll see soon is very important

[00:03:58.960] information because it's telling us how a and b are affecting g through this mathematical expression

[00:04:05.840] so in particular a dot grad is 138 so if we slightly nudge a and make it slightly larger

[00:04:13.840] 138 is telling us that g will grow and the slope of that growth is going to be 138

[00:04:19.040] and the slope of growth of b is going to be 645 so that's going to tell us about how g will respond

[00:04:25.040] if a and b get tweaked a tiny amount in a positive direction okay now you might be confused about

[00:04:32.960] what this expression is that we built out here and this expression by the way is completely

[00:04:37.040] meaningless i just made it up i'm just flexing about the kinds of operations that are supported

[00:04:41.840] by micrograd and what we actually really care about are neural networks but it turns out that

[00:04:46.320] neural networks are just mathematical expressions just like this one but actually a slightly bit

[00:04:51.040] less crazy even neural networks are just a mathematical expression they take the input data

[00:04:57.040] as an input and they take the weights of a neural network as an input and it's a mathematical

[00:05:01.200] expression and the output are your predictions of your neural net or the loss function we'll

[00:05:05.840] see this in a bit but basically neural networks just happen to be a certain class of mathematical

[00:05:10.720] expressions but the back propagation is actually significantly more general it doesn't actually care

[00:05:16.000] about neural networks at all it only tells about arbitrary mathematical expressions and then we

[00:05:20.480] happen to use that machinery for training of neural networks now one more note i would like

[00:05:25.440] to make at this stage is that as you see here micrograd is a scalar valued autograd engine

[00:05:30.240] so it's working on the you know level of individual scalars like negative four and two and we're

[00:05:35.120] taking neural nets and we're breaking them down all the way to these atoms of individual neural

[00:05:39.200] networks and all the little pluses and times and it's just excessive and so obviously you would

[00:05:43.840] never be doing any of this in production it's real just for them for pedagogical reasons because it

[00:05:48.480] allows us to not have to deal with these n-dimensional tensors that you would use in

[00:05:52.800] modern deep neural network library so this is really done so that you understand and refactor

[00:05:58.400] out back propagation and chain rule and understanding of training and then if you actually

[00:06:04.400] want to train bigger networks you have to be using these tensors but none of the math changes this is

[00:06:08.560] done purely for efficiency we are basically taking scale value all the scale values we're

[00:06:13.600] packaging them up into tensors which are just arrays of these scalars and then because we have

[00:06:18.560] these large arrays we're making operations on those large arrays that allows us to take advantage of

[00:06:23.920] the parallelism in a computer and all those operations can be done in parallel and then the

[00:06:28.800] whole thing runs faster but really none of the math changes and they've done purely for efficiency

[00:06:33.040] so i don't think that it's pedagogically useful to be dealing with tensors from scratch and i think

[00:06:37.840] and that's why i fundamentally wrote micrograd because you can understand how things work

[00:06:42.160] at the fundamental level and then you can speed it up later okay so here's the fun part my claim is

[00:06:47.840] that micrograd is what you need to train your own networks and everything else is just efficiency

[00:06:52.400] so you'd think that micrograd would be a very complex piece of code and that turns out to not

[00:06:57.440] be the case so if we just go to micrograd and you'll see that there's only two files here in

[00:07:03.680] micrograd this is the actual engine it doesn't know anything about neural nets and this is the

[00:07:08.320] entire neural nets library on top of micrograd so engine and nn.py so the actual backpropagation

[00:07:16.960] autograd engine that gives you the power of neural networks is literally 100 lines of code

[00:07:24.800] 100 lines of code of like very simple python which we'll understand by the end of this lecture

[00:07:30.640] and then nn.py this neural network library built on top of the autograd engine is like a joke it's

[00:07:38.080] like we have to define what is a neuron and then we have to define what is a layer of neurons and

[00:07:43.360] then we define what is a multi-layer perceptron which is just a sequence of layers of neurons

[00:07:47.840] and so it's just a total joke so basically there's a lot of power that comes from only 150 lines of

[00:07:54.640] code and that's all you need to understand to understand neural network training and everything

[00:07:59.200] else is just efficiency and of course there's a lot to efficiency but fundamentally that's all

[00:08:04.800] that's happening okay so now let's dive right in and implement micrograd step by step the first

[00:08:09.600] thing i'd like to do is i'd like to make sure that you have a very good understanding intuitively of

[00:08:13.440] what a derivative is and exactly what information it gives you so let's start with some basic imports

[00:08:19.440] that i copy-paste in every jupyter notebook always and let's define the function scalar-valued

[00:08:25.280] function f of x as follows so i just made this up randomly i just wanted a scalar-valued function

[00:08:32.080] that takes a single scalar x and returns a single scalar y and we can call this function of course

[00:08:37.680] so we can pass in say 3.0 and get 20 back now we can also plot this function to get a sense of its

[00:08:43.920] shape you can tell from the mathematical expression that this is probably a parabola

[00:08:48.160] it's a quadratic and so if we just create a set of um scale values that we can feed in using for

[00:08:57.440] example a range from negative five to five and steps up on to five so this is so x is just from

[00:09:04.000] negative five to five not including five in steps of 0.25 and we can actually call this function on

[00:09:10.720] this numpy array as well so we get a set of y's if we call f on x's and these y's are basically

[00:09:18.000] also applying uh function on every one of these elements independently and we can plot this using

[00:09:24.640] matplotlib it's a plt.plot x's and y's and we get a nice parabola so previously here we fed in

[00:09:31.280] 3.0 somewhere here and we received 20 back which is here the y coordinate so now i'd like to think

[00:09:37.200] through what is the derivative of this function at any single input point x right so what is the

[00:09:43.520] derivative at different points x of this function now if you remember back to your calculus class

[00:09:48.800] you've probably derived derivatives so we take this mathematical expression 3x squared minus 4x

[00:09:54.080] plus 5 and you would write out on a piece of paper and you would you know apply the product rule and

[00:09:58.000] all the other rules and derive the mathematical expression of the great derivative of the original

[00:10:02.720] function and then you could plug in different x's and see what the derivative is we're not going to

[00:10:07.760] actually do that because no one in neural networks actually writes out the expression for the neural

[00:10:12.960] net it would be a massive expression it would be you know thousands tens of thousands of terms no

[00:10:17.760] one actually derives the derivative of course and so we're not going to take this kind of like

[00:10:22.400] symbolic approach instead what i'd like to do is i'd like to look at the derivative of the

[00:10:25.600] derivative and just make sure that we really understand what the derivative is measuring

[00:10:29.200] what it's telling you about the function and so if we just look up derivative

[00:10:39.200] we see that um okay so this is not a very good definition of derivative this is a definition of

[00:10:43.360] what it means to be differentiable but if you remember from your calculus it is the limit as

[00:10:47.840] h goes to zero of f of x plus h minus f of x over h so basically what it's saying is if you slightly

[00:10:55.760] bump up you're at some point x that you're interested in or hey and if you slightly bump up

[00:11:01.440] you know you slightly increase it by small number h how does the function respond with what

[00:11:06.480] sensitivity does it respond where's the slope at that point does the function go up or does it go

[00:11:11.120] down and by how much and that's the slope of that function the the slope of that response at that

[00:11:17.760] point and so we can basically evaluate um the relative here numerically by taking a very small

[00:11:23.760] h of course the definition would ask us to take h to zero we're just going to pick a very small h

[00:11:28.960] 0.001 and let's say we're interested in 0.3.0 so we can look at f of x of course is 20

[00:11:34.880] and now f of x plus h so if we slightly nudge x in a positive direction how is the function going to

[00:11:40.560] respond and just looking at this do you expand do you expect f of x plus h to be slightly greater

[00:11:46.240] than 20 or do you expect it to be slightly lower than 20 and since this 3 is here and this is 20

[00:11:52.960] if we slightly go positively the function will respond positively so you'd expect this to be

[00:11:58.240] slightly greater than 20 and now by how much is it going to respond to the function

[00:12:02.720] greater than 20 and now by how much is telling you the sort of the the strength of that slope right

[00:12:09.360] the the size of the slope so f of x plus h minus f of x this is how much the function responded

[00:12:16.000] in a positive direction and we have to normalize by the run so we have the rise over run to get

[00:12:22.400] the slope so this of course is just a numerical approximation of the slope because we have to make

[00:12:28.240] h very very small to converge to the exact amount now if i'm doing too many zeros at some point

[00:12:36.800] i'm going to get an incorrect answer because we're using floating point arithmetic and the

[00:12:41.120] representations of all these numbers in computer memory is finite and at some point we get into

[00:12:45.680] trouble so we can converge towards the right answer with this approach but basically um at

[00:12:52.160] 3 the slope is 14 and you can see that by taking 3x square minus 4x plus 5 and differentiating it

[00:12:59.280] in our head so 3x square would be 6x minus 4 and then we plug in x equals 3 so that's 18 minus 4

[00:13:08.000] is 14 so this is correct so that's at 3 now how about the slope at say negative 3 would you expect

[00:13:18.480] what would you expect for the slope now telling the exact value is really hard but what is the

[00:13:23.200] sign of that slope so at negative 3 if we slightly go in the positive direction at x the function

[00:13:30.400] would actually go down and so that tells you that the slope would be negative so we'll get a slight

[00:13:34.560] number below uh below 20 and so if we take the slope we expect something negative negative 22

[00:13:41.920] okay and at some point here of course the slope would be 0 now for this specific function i looked

[00:13:48.640] it up previously and it's at point 2 over 3 so at roughly 2 over 3 that's somewhere here

[00:13:57.040] this derivative would be 0 so basically at that precise point

[00:14:03.440] yeah at that precise point if we nudge in a positive direction the function doesn't respond

[00:14:08.160] this stays the same almost and so that's why the slope is zero okay now let's look at a bit more

[00:14:12.560] complex case so we're going to start you know complexifying a bit so now we have a function

[00:14:19.680] here with output variable d that is a function of three scalar inputs a b and c so a b and c

[00:14:27.200] are some specific values three inputs into our expression graph and a single output d

[00:14:31.920] and so if we just print d we get four and now what i like to do is i'd like to again look at

[00:14:37.680] the derivatives of d with respect to a b and c and think through uh again just the intuition of

[00:14:44.320] what this derivative is telling us so in order to evaluate this derivative we're going to get a bit

[00:14:50.160] hacky here we're going to again have a very small value of h and then we're going to fix the inputs

[00:14:56.320] at some values that we're interested in so these are the this is the point abc at which we're going

[00:15:02.640] to be evaluating the the derivative of d with respect to all a b and c at that point so there's

[00:15:09.040] the inputs and now we have d1 is that expression and then we're going to for example look at the

[00:15:14.160] derivative of d with respect to a so we'll take a and we'll bump it by h and then we'll get d2 to

[00:15:20.480] be the exact same function and now we're going to print um you know f1 d1 is d1 d2 is d2 and print

[00:15:32.640] slope so the derivative or slope here will be um of course d2 minus d1 divided h so d2 minus d1 is

[00:15:44.880] how much the function increased uh when we bumped the uh the specific input that we're interested in

[00:15:52.880] by a tiny amount and this is the normalized by h to get the slope so um yeah so this so

[00:16:06.560] this so i just run this we're going to print d1 which we know is four now d2 will be bumped a will

[00:16:17.920] be bumped by h so let's just think through a little bit uh what d2 will be uh printed out here

[00:16:27.600] in particular d1 will be four will d2 be a number slightly greater than four or slightly lower than

[00:16:35.280] four and that's going to tell us the the sign of the derivative so we're bumping a by h

[00:16:45.440] b is minus three c is 10 so you can just intuitively think through this derivative

[00:16:50.720] and what it's doing a will be slightly more positive and but b is a negative number

[00:16:57.520] so if a is slightly more positive because b is negative three we're actually going to be

[00:17:04.400] adding less to d so you'd actually expect that the value of the function will go down so let's

[00:17:14.160] just see this yeah and so we went from four to 3.9996 and that tells you that the slope will be

[00:17:22.320] negative and then um will be a negative number because we went down and then the exact number

[00:17:29.680] of slope will be exact amount of slope is negative three and you can also convince yourself that

[00:17:35.120] negative three is the right answer mathematically and analytically because if you have a times b

[00:17:39.920] plus c and you are you know you have calculus then uh differentiating a times b plus c with

[00:17:46.000] respect to a gives you just b and indeed the value of b is negative three which is the derivative

[00:17:51.760] that we have so you can tell that that's correct so now if we do this with b so if we bump b by a

[00:17:58.880] little bit in a positive direction we'd get different slopes so what is the influence of b

[00:18:04.400] on the output d so if we bump b by a tiny amount in a positive direction then because a is positive

[00:18:11.600] we'll be adding more to d right so um and now what is the what is the sensitivity what is the

[00:18:17.840] slope of that addition and it might not surprise you that this should be two and why is it two

[00:18:25.280] because d of d by db differentiating with respect to b would be would give us a and the value of a

[00:18:32.720] is two so that's also working well and then if c gets bumped a tiny amount in h by h then of course

[00:18:40.480] a times b is unaffected and now c becomes slightly bit higher what does that do to the function it

[00:18:45.920] makes it slightly bit higher because we're simply adding c and it makes it slightly bit higher by

[00:18:50.480] the exact same amount that we added to c and so that tells you that the slope is one that will be

[00:18:59.200] the rate at which d will increase as we scale c okay so we now have some intuitive sense of

[00:19:06.880] what this derivative is telling you about the function and we'd like to move to neural networks

[00:19:10.960] now as i mentioned neural networks will be pretty massive expressions mathematical expressions

[00:19:15.040] so we need some data structures that maintain these expressions and that's what we're going

[00:19:18.320] to start to build out now so we're going to uh build out this value object that i showed you

[00:19:24.880] in the readme page of micrograd so let me copy paste a skeleton of the first very simple value

[00:19:32.160] object so class value takes a single scalar value that it wraps and keeps track of and that's it so

[00:19:40.800] we can for example do value of 2.0 and then we can get we can look at its content and python

[00:19:48.080] will internally use the repper function to return this string like that so this is a value object

[00:19:59.040] with data equals two that we're creating here now we'd like to do is like we'd like to be able to

[00:20:04.720] have not just like two values but we'd like to do a plus b right we'd like to add them so currently

[00:20:11.840] you would get an error because python doesn't know how to add two value objects so we have to tell it

[00:20:18.720] so here's addition so you have to basically use these special double underscore methods in python

[00:20:27.200] to define these operators for these objects so if we call um the function

[00:20:33.280] if we call um the uh if we use this plus operator python will internally call a dot add of b that's

[00:20:44.000] what will happen internally and so b will be the other and uh self will be a and so we see that

[00:20:51.600] what we're going to return is a new value object and it's just uh it's going to be wrapping the

[00:20:57.040] plus of their data but remember now because uh data is the actual like numbered python number

[00:21:04.160] so this operator here is just the typical floating point plus addition now it's not an addition of

[00:21:10.000] value objects and will return a new value so now a plus b should work and it should print value of

[00:21:17.200] negative one because that's two plus minus three there we go okay let's now implement multiply

[00:21:23.520] just so we can recreate this expression here so multiply i think it won't surprise you

[00:21:28.240] will be fairly similar so instead of add we're going to be using mull and then here of course

[00:21:34.560] we want to do times and so now we can create a c value object which will be 10.0 and now we should

[00:21:40.560] be able to do a times b well let's just do a times b first um that's value of negative six now

[00:21:49.280] and by the way i skipped over this a little bit uh suppose that i didn't have the repper function

[00:21:52.960] here uh then it's just that you'll get some kind of an ugly expression so what repper is doing is

[00:21:59.120] it's providing us a way to print out like a nicer looking expression in python uh so we don't just

[00:22:04.800] have something cryptic we actually are you know it's value of negative six so this gives us a

[00:22:11.280] times and then this we should now be able to add c to it because we've defined and told the python

[00:22:17.120] because we've defined and told the python how to do mull and add and so this will call this will

[00:22:22.400] basically be equivalent to a dot mull of b and then this new value object will be dot add of c

[00:22:32.560] and so let's see if that worked yep so that worked well that gave us four which is what

[00:22:37.200] we expect from before and i believe we can just call them manually as well there we go so

[00:22:42.800] yeah okay so now what we are missing is the connected tissue of this expression as i mentioned

[00:22:47.680] we want to keep these expression graphs so we need to know and keep pointers about what values

[00:22:53.040] produce what other values so here for example we are going to introduce a new variable which we'll

[00:22:58.480] call children and by default it will be an empty tuple and then we're actually going to keep a

[00:23:03.040] slightly different variable in the class which we'll call underscore prev which will be the set

[00:23:07.920] of children uh this is how i done i did it in the original micrograd looking at my code here

[00:23:13.920] i can't remember exactly the reason i believe it was efficiency but this underscore children will

[00:23:18.320] be a tuple for convenience but then when we actually maintain it in the class it will be

[00:23:21.920] just this set yes i i believe for efficiency um so now when we are creating a value like this

[00:23:30.160] with a constructor children will be empty and prev will be the empty set but when we are creating

[00:23:35.040] a value through addition or multiplication we're going to feed in the children of this

[00:23:40.160] value which in this case is self and other so those are the children here

[00:23:49.040] so now we can do d.prev and we'll see that the children of the we now know are this a value of

[00:23:56.480] negative six and value of ten and this of course is the value resulting from a times b and the c

[00:24:02.640] value which is ten now the last piece of information we don't know so we know now the

[00:24:08.320] children of every single value but we don't know what operation created this value so we need one

[00:24:13.520] more element here let's call it underscore pop and by default this is the empty set for leaves

[00:24:20.480] and then we'll just maintain it here and now the operation will be just a simple string

[00:24:26.400] and in the case of addition it's plus in the case of multiplication is times so now we not

[00:24:33.520] just have d.prev we also have a d.op and we know that d was produced by an addition of those two

[00:24:39.520] values and so now we have the full mathematical expression and we're building out this data

[00:24:44.880] structure and we know exactly how each value came to be by what expression and from what other values

[00:24:50.000] now because these expressions are about to get quite a bit larger we'd like a way to nicely

[00:24:54.160] visualize these expressions that we're building out so for that i'm going to copy paste a bunch of

[00:24:59.360] slightly scary code that's going to visualize this these expression graphs for us so here's

[00:25:04.880] the code and i'll explain it in a bit but first let me just show you what this code does basically

[00:25:10.160] what it does is it creates a new function draw dot that we can call on some root node

[00:25:15.040] and then it's going to visualize it so if we call draw dot on d which is this final value here that

[00:25:21.200] is a times b plus c it creates something like this so this is d and you see that this is a times b

[00:25:29.280] creating an interpretive value plus c gives us this output node d so that's drawn out of d and

[00:25:37.440] i'm not going to go through this in complete detail but i'm going to show you how to do that

[00:25:41.120] right out of d and i'm not going to go through this in complete detail you can take a look at

[00:25:46.240] graphvis and its api graphvis is a open source graph visualization software and what we're doing

[00:25:52.080] here is we're building out this graph in graphvis api and you can basically see that trace is this

[00:25:58.960] helper function that enumerates all the nodes and edges in the graph so that just builds a

[00:26:03.760] set of all the nodes and edges and then we iterate for all the nodes and we create special node

[00:26:08.080] objects for them in using dot node and then we also create edges using dot dot edge

[00:26:16.880] and the only thing that's like slightly tricky here is you'll notice that i

[00:26:20.480] basically add these fake nodes which are these operation nodes so for example this node here

[00:26:25.760] is just like a plus node and i create these special op nodes here and i connect them

[00:26:35.760] accordingly so these nodes of course are not actual nodes in the original graph they're not

[00:26:42.160] actually a value object the only value objects here are the things in squares those are actual

[00:26:47.840] value objects or representations thereof and these op nodes are just created in this draw dot

[00:26:53.200] routine so that it looks nice let's also add labels to these graphs just so we know what

[00:26:58.240] variables are where so let's create a special underscore label um or let's just do label equals

[00:27:06.160] empty by default and save it in each node and then here we're going to do label as a

[00:27:13.520] a label is b label is c um and then let's create a special um equals a times b

[00:27:30.800] and e dot label will be e it's kind of naughty and e will be e plus c and a d dot label will be

[00:27:40.000] b okay so nothing really changes i just added this new e function a new e variable and then here when

[00:27:48.720] we are printing this i'm going to print the label here so this will be a percent s bar and this will

[00:27:56.560] be n dot label and so now we have the label on the left here so it says a b creating e and then

[00:28:06.160] e plus c creates d just like we have it here and finally let's make this expression just one layer

[00:28:12.240] deeper so d will not be the final output node instead after d we are going to create a new value

[00:28:19.600] object called f we're going to start running out of variables soon f will be negative 2.0 and its

[00:28:26.880] label will of course just be f and then l capital l will be the output of our graph and l will be

[00:28:35.360] and l will be d times f okay so l will be negative 8 is the output so now we don't just draw a d we

[00:28:46.880] draw l okay and somehow the label of l is undefined oops although label has to be explicitly sort of

[00:28:57.920] given to it there we go so l is the output so let's quickly recap what we've done so far

[00:29:04.080] we are able to build out mathematical expressions using only plus and times so far

[00:29:09.200] they are scalar valued along the way and we can do this forward pass and build out a mathematical

[00:29:15.440] expression so we have multiple inputs here a b c and f going into a mathematical expression that

[00:29:21.680] produces a single output l and this here is visualizing the forward pass so the output of

[00:29:27.920] the forward pass is negative 8 that's the value now what we'd like to do next is we'd like to

[00:29:33.760] run back propagation and in back propagation we are going to start here at the end and we're going

[00:29:39.120] to reverse and calculate the gradient along along all these intermediate values and really what

[00:29:46.000] we're computing for every single value here we're going to compute the derivative of that

[00:29:51.920] node with respect to l so the derivative of l with respect to l is just one and then we're going to

[00:30:01.200] derive what is the derivative of l with respect to f with respect to d with respect to c with

[00:30:06.400] respect to e with respect to b and with respect to a and in a neural network setting you'd be

[00:30:12.320] very interested in the derivative of basically this loss function l with respect to the weights

[00:30:18.080] of a neural network and here of course we have just these variables a b c and f but some of

[00:30:23.040] these will eventually represent the weights of a neural net and so we'll need to know how those

[00:30:27.360] weights are impacting the loss function so we'll be interested basically in the derivative of the

[00:30:32.320] output with respect to some of its leaf nodes and those leaf nodes will be the weights of the neural

[00:30:37.360] net and the other leaf nodes of course will be the data itself but usually we will not want or

[00:30:42.640] use the derivative of the loss function with respect to data because the data is fixed but

[00:30:47.600] the weights will be iterated on using the gradient information so next we are going to create a

[00:30:53.440] variable inside the value class that maintains the derivative of l with respect to that value

[00:31:01.040] and we will call this variable grad so there is a dot data and there's a self at grad and initially

[00:31:08.240] it will be zero and remember that zero is basically means no effect so at initialization

[00:31:14.160] we're assuming that every value does not impact does not affect the out the output right because

[00:31:20.240] if the gradient is zero that means that changing this variable is not changing the loss function

[00:31:25.680] so by default we assume that the gradient is zero and then now that we have grad and it's 0.0

[00:31:36.640] we are going to be able to visualize it here after data so here grad is 0.4f

[00:31:41.360] and this will be in that graph and now we are going to be showing both the data and the grad

[00:31:50.080] initialized at zero and we are just about getting ready to calculate the back propagation

[00:31:56.480] and of course this grad again as i mentioned is representing the derivative of the output

[00:32:01.040] in this case l with respect to this value so with respect to so this is the derivative of

[00:32:06.400] l with respect to f with respect to d and so on so let's now fill in those gradients and actually

[00:32:11.520] do back propagation manually so let's start filling in these gradients and start all the way at the

[00:32:15.760] end as i mentioned here first we are interested to fill in this gradient here so what is the derivative

[00:32:21.520] of l with respect to l in other words if i change l by a tiny amount h how much does l change it

[00:32:31.440] changes by h so it's proportional and therefore the derivative will be one we can of course

[00:32:38.160] measure these or estimate these numerical gradients numerically just like we've seen before so if i

[00:32:43.840] take this expression and i create a def lol function here and put this here now the reason

[00:32:51.040] i'm creating a gating function lol here is because i don't want to pollute or mess up the global scope

[00:32:56.160] here this is just kind of like a little staging area and as you know in python all of these will

[00:33:00.400] be local variables to this function so i'm not changing any of the global scope here so here l1

[00:33:06.720] will be l and then copy pasting this expression we're going to add a small amount h in for example

[00:33:18.640] a right and this would be measuring the derivative of l with respect to a so here this will be l2

[00:33:27.040] and then we want to print that derivative so print l2 minus l1 which is how much l changed

[00:33:33.840] and then normalize it by h so this is the rise over run and we have to be careful because l

[00:33:39.840] is a value node so we actually want its data so that these are floats dividing by h and this should

[00:33:48.400] print the derivative of l with respect to a because a is the one that we bumped a little bit by h so

[00:33:54.640] what is the derivative of l with respect to a it's six okay and obviously if we change l

[00:34:05.040] by h then that would be here effectively this looks really awkward but changing l by h

[00:34:16.080] you see the derivative here is one

[00:34:17.680] that's kind of like the base case of what we are doing here so basically we can come up here and

[00:34:23.680] we can manually set l.grad to one this is our manual backpropagation l.grad is one and let's

[00:34:30.480] redraw and we'll see that we filled in grad is one for l we're now going to continue the

[00:34:36.880] backpropagation so let's here look at the derivatives of l with respect to d and f let's do

[00:34:43.040] a d first so what we are interested in if i create a markdown on here is we'd like to know

[00:34:48.960] basically we have that l is d times f and we'd like to know what is uh d l by d d what is that

[00:34:58.640] and if you know your calculus uh l is d times f so what is d l by d d it would be f and if you

[00:35:05.920] don't believe me we can also just derive it because the proof would be fairly straightforward

[00:35:09.920] we go to the definition of the derogative which is f of x plus h minus f of x divide h

[00:35:18.320] as a limit limit of h goes to zero of this kind of expression so when we have l is d times f then

[00:35:25.040] increasing d by h would give us the output of d plus h times f that's basically f of x plus h

[00:35:33.520] right minus d times f and then divide h and symbolically expanding out here we would have

[00:35:42.560] basically d times f plus h times f minus d times f divide h and then you see how the df minus df

[00:35:50.480] cancels so you're left with h times f divide h which is f so in the limit as h goes to zero of

[00:35:58.400] you know um derogative um definition we just get f in the case of d times f

[00:36:07.440] so symmetrically d l by d f will just be d so what we have is that f dot grad we see now

[00:36:17.440] is just the value of d which is four

[00:36:20.320] and we see that d dot grad is just uh the value of f

[00:36:28.560] and so the value of f is negative two so we'll set those manually

[00:36:36.720] let me erase this markdown node and then let's redraw what we have

[00:36:40.080] okay and let's just make sure that these were correct so we seem to think that d l by d d is

[00:36:45.920] negative two so let's double check um let me erase this plus h from before and now we want

[00:36:51.840] the derivative with respect to f so let's just come here when i create f and let's do a plus h

[00:36:57.040] here and this should print a derivative of f plus h so let's do a plus h here and let's

[00:37:03.120] do a plus h here and this should print a derivative of l with respect to f so we expect to see four

[00:37:10.640] yeah and this is four up to floating point funkiness and then d l by d d should be f

[00:37:19.200] which is negative two grad is negative two so if we again come here and we change d

[00:37:25.920] d dot data plus equals h right here so we expect so we've added a little h and then we see how l

[00:37:32.800] changed and we expect to print uh negative two there we go so we've numerically verified what

[00:37:43.440] we're doing here is kind of like an inline gradient check gradient check is when we are

[00:37:48.640] deriving this like background check and then we're going to print a negative two

[00:37:53.040] check is when we are deriving this like back propagation and getting the derivative with

[00:37:57.440] respect to all the intermediate results and then numerical gradient is just you know

[00:38:03.200] estimating it using small step size now we're getting to the crux of back propagation so this

[00:38:09.440] will be the most important node to understand because if you understand the gradient for this

[00:38:14.400] node you understand all of back propagation and all of training of neural nets basically so we

[00:38:20.000] need to derive dl by dc in other words the derivative of l with respect to c because

[00:38:26.880] we've computed all these other gradients already now we're coming here and we're continuing the

[00:38:31.200] back propagation manually so we want dl by dc and then we'll also derive dl by de now here's

[00:38:38.800] a problem how do we derive dl by dc we actually know the derivative l with respect to d so we know

[00:38:47.760] how l is sensitive to d but how is l sensitive to c so if we wiggle c how does that impact l through

[00:38:56.000] d so we know dl by dc and we also here know how c impacts d and so just very intuitively if you

[00:39:06.160] know the impact that c is having on d and the impact that d is having on l then you should be

[00:39:11.920] able to somehow put that information together to figure out how c impacts l and indeed this is what

[00:39:17.440] we can actually do so in particular we know just concentrating on d first let's look at how what is

[00:39:23.920] the derivative basically of d with respect to c so in other words what is dd by dc

[00:39:31.600] so here we know that d is c times c plus e that's what we know and now we're interested in dd by dc

[00:39:39.600] if you just know your calculus again and you remember then differentiating c plus e with

[00:39:44.000] respect to c you know that that gives you 1.0 and we can also go back to the basics and derive this

[00:39:50.880] because again we can go to our f of x plus h minus f of x divided by h that's the definition

[00:39:57.280] of a derivative as h goes to zero and so here focusing on c and its effect on d we can basically

[00:40:05.120] do the f of x plus h will be c is incremented by h plus c that's the first evaluation of our function

[00:40:12.640] minus c plus e and then divide h and so what is this just expanding this out this will be c plus h

[00:40:22.640] plus c minus c minus e divide h and then you see here how c minus c cancels e minus e cancels we're

[00:40:30.320] left with h over h which is 1.0 and so by symmetry also d d by d e will be 1.0 as well so basically

[00:40:43.440] the derivative of a sum expression is very simple and this is the local derivative so i call this

[00:40:48.960] the local derivative because we have the final output value all the way at the end of this graph

[00:40:53.520] and we're now like a small node here and this is a little plus node and the little plus node

[00:40:59.680] doesn't know anything about the rest of the graph that it's embedded in all it knows is that it did

[00:41:04.720] plus it took a c and an e added them and created a d and this plus node also knows the local

[00:41:11.280] influence of c on d or rather the derivative of d with respect to c and it also knows the

[00:41:17.440] derivative of d with respect to e but that's not what we want that's just a local derivative what

[00:41:22.960] we actually want is dl by dc and l could l is here just one step away but in a general case

[00:41:30.720] this little plus node is could be embedded in like a massive graph so again we know how l impacts d

[00:41:37.920] and now we know how c and e impact d how do we put that information together to write dl by dc

[00:41:44.240] and the answer of course is the chain rule in calculus and so i pulled up a chain rule here

[00:41:51.360] from Wikipedia and i'm going to go through this very briefly so chain rule Wikipedia sometimes

[00:41:58.080] can be very confusing and calculus can be very confusing like this is the way i learned chain

[00:42:05.280] rule and it was very confusing like what is happening it's just complicated so i like this

[00:42:10.320] expression much better if a variable z depends on a variable y which itself depends on a variable x

[00:42:16.960] then z depends on x as well obviously through the intermediate variable y in this case the chain

[00:42:22.400] rule is expressed as if you want dz by dx then you take the dz by dy and you multiply it by dy

[00:42:31.200] by dx so the chain rule fundamentally is telling you how we chain these derivatives together

[00:42:39.680] derivatives together correctly so to differentiate through a function composition we have to apply

[00:42:47.200] a multiplication of those derivatives so that's really what chain rule is telling us and there's

[00:42:55.280] a nice little intuitive explanation here which i also think is kind of cute the channel states

[00:42:59.840] that knowing the instantaneous rate of change of z with respect to y and y relative to x allows

[00:43:04.240] one to calculate the instantaneous rate of change of z relative to x as a product of those two rates

[00:43:09.440] of change simply the product of those two so here's a good one if a car travels twice as

[00:43:15.600] fast as a bicycle and the bicycle is four times as fast as a walking man then the car travels

[00:43:21.200] two times four eight times as fast as a man and so this makes it very clear that the correct thing

[00:43:27.840] to do sort of is to multiply so car is twice as fast as bicycle and bicycle is four times as fast

[00:43:35.600] as man so the car will be eight times as fast as the man and so we can take these intermediate

[00:43:42.720] rates of change if you will and multiply them together and that justifies the chain rule

[00:43:49.120] intuitively so have a look at chain rule about here really what it means for us is there's a

[00:43:54.240] very simple recipe for deriving what we want which is dl by dc and what we have so far is we know

[00:44:02.240] want and we know what is the impact of d on l so we know dl by dd the derivative of l with respect

[00:44:13.760] to dd we know that that's negative two and now because of this local reasoning that we've done

[00:44:19.360] here we know dd by dc so how does c impact d and in particular this is a plus node so the local

[00:44:28.880] derivative is simply 1.0 it's very simple and so the chain rule tells us that dl by dc going

[00:44:36.480] through this intermediate variable will just be simply dl by dd times

[00:44:47.920] dd by dc that's chain rule so this is identical to what's happening here except

[00:44:56.240] z is rl y is rd and x is rc so we literally just have to multiply these and because

[00:45:08.400] these local derivatives like dd by dc are just one we basically just copy over dl by dd because

[00:45:15.600] this is just times one so what is it so because dl by dd is negative two what is dl by dc well

[00:45:24.080] it's the local gradient 1.0 times dl by dd which is negative two so literally what a plus node does

[00:45:31.280] you can look at it that way is it literally just routes the gradient because the plus nodes local

[00:45:36.720] derivatives are just one and so in the chain rule one times dl by dd is just dl by dd and so that

[00:45:48.080] derivative just gets routed to both c and to e in this case so basically we have that e.grad

[00:45:57.040] or let's start with c since that's the one we looked at is negative two times one negative two

[00:46:06.160] and in the same way by symmetry e.grad will be negative two that's the claim so we can set those

[00:46:12.960] we can redraw and you see how we just assign negative two negative two so this back propagating

[00:46:19.120] signal which is carrying the information of like what is the derivative of l with respect to all

[00:46:23.200] the intermediate nodes we can imagine it almost like flowing backwards through the graph and a

[00:46:28.000] plus node will simply distribute the derivative to all the leaf nodes sorry to all the children

[00:46:33.040] nodes of it so this is the claim and now let's verify it so let me remove the plus node and

[00:46:39.440] now instead what we're going to do is we want to increment c so c.data will be incremented by h

[00:46:44.800] and when i run this we expect to see negative two negative two and then of course for e

[00:46:53.200] so e.data plus equals h and we expect to see negative two simple

[00:46:57.680] so those are the derivatives of these internal nodes and now we're going to recurse our way

[00:47:07.840] backwards again and we're again going to apply the chain rule so here we go our second application

[00:47:13.840] of chain rule and we will apply it all the way through the graph we just happen to only have

[00:47:17.840] one more node remaining we have that dl by d and we're going to apply the chain rule

[00:47:23.680] one more node remaining we have that dl by de as we have just calculated is negative two

[00:47:30.720] so we know that so we know the derivative of l with respect to e

[00:47:36.720] and now we want dl by da right and the chain rule is telling us that that's just dl by de

[00:47:46.240] negative two times the local gradient so what is the local gradient basically de by da we have to

[00:47:54.800] look at that so i'm a little times node inside a massive graph and i only know that i did a times b

[00:48:03.760] and i produced an e so now what is de by da and de by db that's the only thing that i sort of

[00:48:11.120] know about that's my local gradient so because we have that e is a times b we're asking what is de

[00:48:18.800] by da and of course we just did that here we had a times so i'm not going to re-derive it

[00:48:26.560] but if you want to differentiate this with respect to a you'll just get b right the value of b which

[00:48:33.600] in this case is negative 3.0 so basically we have that dl by da well let me just do it right here

[00:48:43.680] we have that a dot grad and we are applying chain rule here is dl by de which we see here is

[00:48:50.800] negative two times what is de by da it's the value of b which is negative three

[00:48:59.440] that's it and then we have b dot grad is again dl by de which is negative two just the same way

[00:49:08.800] times what is de by d um db is the value of a which is 2.0 that's the value of a

[00:49:19.120] so these are our claimed derivatives let's redraw and we see here that a dot grad turns out to be

[00:49:28.720] six because that is negative two times negative three and b dot grad is negative four times sorry

[00:49:35.760] is negative two times two which is negative four so those are our claims let's delete this and

[00:49:41.360] let's verify them we have a here a dot data plus equals h so the claim is that a dot grad is six

[00:49:55.120] let's verify six and we have b dot data plus equals h so nudging b by h and then we have

[00:50:05.600] b dot data plus equals h so nudging b by h and looking at what happens we claim it's negative four

[00:50:15.120] and indeed it's negative four plus minus again float oddness and that's it this that was the

[00:50:25.360] manual back propagation all the way from here to all the leaf nodes and we've done it piece by

[00:50:32.560] piece and really all we've done is as you saw we iterated through all the nodes one by one

[00:50:37.840] and locally applied the chain rule we always know what is the derivative of l with respect to this

[00:50:43.040] little output and then we look at how this output was produced this output was produced through some

[00:50:48.080] operation and we have the pointers to the children nodes of this operation and so in this little

[00:50:53.760] operation we know what the local derivatives are and we just multiply them onto the derivative

[00:50:58.560] always so we just go through and recursively multiply on the local derivatives and that's

[00:51:04.320] what back propagation is it's just a recursive application of chain rule backwards through the

[00:51:08.880] computation graph let's see this power in action just very briefly what we're going to do is we're

[00:51:14.560] going to nudge our inputs to try to make l go up so in particular what we're doing is we want a

[00:51:22.000] data we're going to change it and if we want l to go up that means we just have to go in the

[00:51:27.280] direction of the gradient so a should increase in the direction of gradient by like some small

[00:51:33.920] step amount this is the step size and we don't just want this for b but also for b

[00:51:41.440] also for c also for f those are leaf nodes which we usually have control over and if we nudge in

[00:51:52.000] direction of the gradient we expect a positive influence on l so we expect l to go up positively

[00:51:59.920] so it should become less negative it should go up to say negative you know six or something like

[00:52:04.560] that it's hard to tell exactly and we have to rerun the forward pass so let me just do that here

[00:52:16.560] this would be the forward pass f would be unchanged this is effectively the forward pass

[00:52:21.840] and now if we print l dot data we expect because we nudged all the values all the inputs in the

[00:52:28.480] rational gradient we expected a less negative l we expect it to go up so maybe it's negative

[00:52:34.080] six or so let's see what happens okay negative seven and this is basically one step of an

[00:52:41.840] optimization that will end up running and really this gradient just give us some power because we

[00:52:47.200] know how to influence the final outcome and this will be extremely useful for training

[00:52:50.880] neural nets as well as cnc so now i would like to do one more example of manual back propagation

[00:52:57.760] using a bit more complex and useful example we are going to back propagate through a neuron

[00:53:05.440] so we want to eventually build out neural networks and in the simplest case these are

[00:53:10.960] multi-layer perceptrons as they're called so this is a two layer neural net and it's got these hidden

[00:53:16.480] layers made up of neurons and these neurons are fully connected to each other now biologically

[00:53:20.800] neurons are very complicated devices but we have very simple mathematical models of them

[00:53:26.080] and so this is a very simple mathematical model of a neuron you have some inputs x's and then you

[00:53:32.160] have these synapses that have weights on them so the w's are weights and then the synapse interacts

[00:53:41.920] with the input to this neuron multiplicatively so what flows to the cell body of this neuron

[00:53:48.080] is w times x but there's multiple inputs so there's many w times x's flowing to the cell body

[00:53:54.960] the cell body then has also like some bias so this is kind of like the

[00:53:59.680] inner innate sort of trigger happiness of this neuron so this bias can make it a bit more trigger

[00:54:05.280] happy or a bit less trigger happy regardless of the input but basically we're taking all the w

[00:54:10.160] times x of all the inputs adding the bias and then we take it through an activation function

[00:54:16.000] and this activation function is usually some kind of a squashing function like a sigmoid or 10h or

[00:54:21.040] something like that so as an example we're going to use the 10h in this example numpy has a np dot

[00:54:29.280] 10h so we can call it on a range then we can plot it this is the 10h function and you see that the

[00:54:38.160] inputs as they come in get squashed on the y coordinate here so right at zero we're going to

[00:54:45.760] get exactly zero and then as you go more positive in the input then you'll see that the function

[00:54:51.200] will only go up to one and then plateau out and so if you pass in very positive inputs we're going

[00:54:57.440] to cap it smoothly at one and on the negative side we're going to cap it smoothly to negative one

[00:55:02.560] so that's 10h and that's the squashing function or an activation function and what comes out of

[00:55:08.800] this neuron is just the activation function applied to the dot product of the weights and the inputs

[00:55:16.000] so let's write one out i'm going to copy paste because

[00:55:22.640] i'm going to copy paste because i don't want to type too much but okay so here we have the inputs

[00:55:31.520] x1 x2 so this is a two-dimensional neuron so two inputs are going to come in these are thought of

[00:55:36.800] as the weights of this neuron weights w1 w2 and these weights again are the synaptic strengths

[00:55:43.520] for each input and this is the bias of the neuron b and now we want to do is according to this model

[00:55:52.160] we need to multiply x1 times w1 and x2 times w2 and then we need to add bias on top of it

[00:56:01.440] and it gets a little messy here but all we are trying to do is x1 w1 plus x2 w2 plus b

[00:56:07.920] and these are multiplied here except i'm doing it in small steps so that we actually

[00:56:12.800] have pointers to all these intermediate nodes so we have x1 w1 variable x times x2 w2 variable

[00:56:19.360] and i'm also labeling them so n is now the cell body raw activation without

[00:56:28.400] the activation function for now and this should be enough to basically plot it so draw a dot of n

[00:56:37.840] gives us x1 times w1 x2 times w2 being added then the bias gets added on top of this

[00:56:45.040] and this n is this sum so we're now going to take it through an activation function

[00:56:52.000] and let's say we use the tanh so that we produce the output so what we'd like to do here is we'd

[00:56:57.520] like to do the output and i'll call it o is n dot tanh okay but we haven't yet written the tanh

[00:57:07.760] now the reason that we need to implement another tanh function here is that tanh is a hyperbolic

[00:57:14.800] function and we've only so far implemented a plus and a times and you can't make a tanh

[00:57:19.520] out of just pluses and times you also need exponentiation so tanh is this kind of a formula

[00:57:25.440] here you can use either one of these and you see that there is exponentiation involved which we

[00:57:30.240] have not implemented yet for our little value node here so we're not going to be able to produce

[00:57:35.040] tanh yet and we have to go back up and implement something like it now one option here is we could

[00:57:42.560] actually implement exponentiation right and we could return the exp of a value instead of a tanh

[00:57:50.560] of a value because if we had exp then we have everything else that we need so because we know

[00:57:56.880] how to add and we know how to we know how to add and we know how to multiply so we'd be able to

[00:58:03.280] create tanh if we knew how to exp but for the purposes of this example i specifically wanted to

[00:58:08.720] show you that we don't necessarily need to have the most atomic pieces in in this value object

[00:58:16.960] we can actually like create functions at arbitrary points of abstraction they can be complicated

[00:58:24.080] functions but they can be also very very simple functions like a plus and it's totally up to us

[00:58:28.720] the only thing that matters is that we know how to differentiate through any one function so we

[00:58:33.840] take some inputs and we make an output the only thing that matters it can be arbitrarily complicated

[00:58:38.000] arbitrarily complex function as long as you know how to create the local derivative if you know the

[00:58:43.360] local derivative of how the inputs impact the output then that's all you need so we're going

[00:58:47.760] to cluster up all of this expression and we're not going to break it down to its atomic pieces

[00:58:53.200] we're just going to directly implement tanh so let's do that dab tanh and then out will be a value

[00:59:00.720] of and we need this expression here so um let me actually copy paste

[00:59:14.160] let's grab n which is a cell dot theta and then this i believe is the tanh math dot exp of

[00:59:22.560] two no and my n minus one over two and plus one maybe i can call this x just so that it matches

[00:59:32.000] exactly okay and now this will be t and uh children of this node there's just one child

[00:59:42.080] and i'm wrapping it in a tuple so this is a tuple of one object just self and here the name of this

[00:59:48.400] operation will be tanh and we're going to return that okay so now values should be implementing tanh

[00:59:59.920] and now we can scroll all the way down here and we can actually do n dot tanh and that's going to

[01:00:05.200] return the tanh output of n and now we should be able to draw it out of o not of n so let's see how

[01:00:13.600] that worked there we go and went through tanh to produce this output so now tanh is a sort of

[01:00:26.000] our little micro grad supported node here as an operation and as long as we know the derivative

[01:00:33.120] of tanh then we'll be able to back propagate through it now let's see this tanh in action

[01:00:38.240] currently it's not squashing too much because the input to it is pretty low so the bias was increased

[01:00:44.400] to say eight then we'll see that what's flowing into the tanh now is two and tanh is squashing

[01:00:53.600] it to 0.96 so we're already hitting the tail of this tanh and it will sort of smoothly go up to

[01:00:59.200] one and then plateau out over there okay so now i'm going to do something slightly strange i'm

[01:01:04.000] going to change this bias from eight to this number 6.88 etc and i'm going to do this for

[01:01:10.880] specific reasons because we're about to start back propagation and i want to make sure that

[01:01:16.400] our numbers come out nice they're not like very crazy numbers they're nice numbers that we can

[01:01:20.640] sort of understand in our head let me also add o's label o is short for output here

[01:01:26.560] so that's the error okay so 0.88 flows into tanh comes out 0.7 so on so now we're going to do back

[01:01:33.440] propagation and we're going to fill in all the gradients so what is the derivative o with respect

[01:01:39.440] to all the inputs here and of course in a typical neural network setting what we really care about

[01:01:45.200] the most is the derivative of these neurons on the weights specifically the w2 and w1 because

[01:01:52.400] those are the weights that we're going to be changing part of the optimization and the other

[01:01:56.320] thing that we have to remember is here we have only a single neuron but in the neural net you

[01:01:59.920] typically have many neurons and they're connected so this is only like a one small neuron a piece

[01:02:05.520] of a much bigger puzzle and eventually there's a loss function that sort of measures the accuracy

[01:02:09.760] of the neural net and we're back propagating with respect to that accuracy and trying to increase it

[01:02:15.280] so let's start off by propagation here and end so let's start off with back propagation

[01:02:20.960] here in the end what is the derivative of o with respect to o the base case sort of we know always

[01:02:27.440] is that the gradient is just 1.0 so let me fill it in and then uh let me split out uh the drawing

[01:02:37.760] function um here and then here cell clear this output here okay so now when we draw o we'll see

[01:02:51.920] that oh that grad is one so now we're going to back propagate through the 10h so to back propagate

[01:02:57.840] through 10h we need to know the local derivative of 10h so if we have that uh o is 10h of n then

[01:03:08.640] what is d o by d n now what you could do is you could come here and you could take this expression

[01:03:15.360] and you could do your calculus derivative taking um and that would work but we can also just scroll

[01:03:21.840] down wikipedia here into a section that hopefully tells us that derivative uh d by dx of 10h of x

[01:03:30.560] is any of these i like this one one minus 10h square of x so this is one minus 10h of x squared

[01:03:39.360] so basically what this is saying is that d o by dn is one minus 10h of n squared

[01:03:49.280] and we already have 10h of n it's just o so it's one minus o squared so o is the output here so

[01:03:56.480] the output is this number o dot data is this number and then what this is saying is that d o

[01:04:07.520] by dn is one minus this squared so one minus o dot data squared is 0.5 conveniently so that's

[01:04:17.920] the local derivative of this 10h operation here is 0.5 and uh so that would be d o by dn

[01:04:25.840] so we can fill in that n dot grad is 0.5 we'll just fill it in

[01:04:37.120] so this is exactly 0.5 one half so now we're going to continue the back propagation

[01:04:49.280] this is 0.5 and this is a plus node so how is backdrop going to what is backdrop going to do

[01:04:55.680] here and if you remember our previous example a plus is just a distributor of gradient so this

[01:05:02.160] gradient will simply flow to both of these equally and that's because the local derivative of this

[01:05:06.960] operation is one for every one of its nodes so one times 0.5 is 0.5 so therefore we know that

[01:05:14.960] this node here which we called this its grad is just 0.5 and we know that b dot grad is also 0.5

[01:05:24.880] so let's set those and let's draw

[01:05:28.880] so those are 0.5 continuing we have another plus 0.5 again we'll just distribute so 0.5 will flow

[01:05:35.840] to both of these so we can set theirs x2w2 as well dot grad is 0.5 and let's redraw pluses

[01:05:49.520] are my favorite operations to back propagate through because it's very simple so now what's

[01:05:55.520] flowing into these expressions is 0.5 and so really again keep in mind what the derivative

[01:05:59.520] is telling us at every point in time along here this is saying that if we want the output of this

[01:06:05.600] neuron to increase then the influence on these expressions is positive on the output both of them

[01:06:13.200] are positive contribution to the output so now back propagating to x2 and w2 first this is a

[01:06:24.640] times node so we know that the local derivative is you know the other term so if we want to

[01:06:29.920] calculate x2 dot grad then can you think through what it's going to be so x2 dot grad will be w2

[01:06:43.440] dot data times this x2w2 dot grad right and w2 dot grad will be x2 dot data times x2w2 dot grad

[01:07:01.360] right so that's the little local piece of chain rule

[01:07:07.040] let's set them and let's redraw so here we see that the gradient on our weight

[01:07:11.840] two is zero because x2's data was zero right but x2 will have the gradient 0.5 because data here

[01:07:19.440] was one and so what's interesting here right is because the input x2 was zero then because

[01:07:25.840] of the way the times works of course this gradient will be zero and think about intuitively why that

[01:07:31.760] is derivative always tells us the influence of this on the final output if i wiggle w2 how is

[01:07:40.080] the output changing it's not changing because we're multiplying by zero so because it's not

[01:07:45.200] changing there is no derivative and zero is the correct answer because we're squashing that zero

[01:07:52.160] and let's do it here 0.5 should come here and flow through this times and so we'll have

[01:07:58.960] that x1 dot grad is can you think through a little bit what what this should be

[01:08:05.040] the local derivative of times with respect to x1 is going to be w1 so w1's data times x1w1

[01:08:14.960] dot grad and w1 dot grad will be x1 dot data times x1w2 w1 dot grad let's see what those

[01:08:25.760] came out to be so this is 0.5 so this would be negative 1.5 and this would be 1 and this

[01:08:33.440] 1 and we've back propagated through this expression these are the actual final derivatives

[01:08:39.120] so if we want this neuron's output to increase we know that what's necessary is that w2 we have no

[01:08:48.080] gradient w2 doesn't actually matter to this neuron right now but this neuron this weight

[01:08:53.360] should go up so if this weight goes up then this neuron's output would have gone up and

[01:08:59.680] proportionally because the gradient is one okay so doing the back propagation manually is obviously

[01:09:04.480] ridiculous so we are now going to put an end to this suffering and we're going to see how we can

[01:09:09.200] implement the backward pass a bit more automatically we're not going to be doing all of

[01:09:13.280] it manually out here it's now pretty obvious to us by example how these pluses and times are back

[01:09:18.560] propagating gradients so let's go up to the value object and we're going to start codifying what

[01:09:25.520] we've seen in the examples below so we're going to do this by storing a special self dot backward

[01:09:34.800] and underscore backward and this will be a function which is going to do that little

[01:09:39.680] piece of chain rule at each little node that compute that took inputs and produced output

[01:09:45.200] we're going to store how we are going to chain the the outputs gradient into the inputs gradients

[01:09:51.440] so by default this will be a function that doesn't do anything so and you can also see

[01:09:59.760] that here in the value in micro grad so we have this backward function by default doesn't do

[01:10:06.080] anything this is a empty function and that would be sort of the case for example for a leaf node

[01:10:11.600] for leaf node there's nothing to do but now if when we're creating these out values these out

[01:10:18.720] values are an addition of self and other and so we will want to sell set outs backward to be

[01:10:28.560] the function that propagates the gradient so let's define what should happen

[01:10:39.600] and we're going to store it in the closure let's define what should happen when we call outs grad

[01:10:44.800] for addition our job is to take outs grad and propagate it into selfs grad and other dot grad

[01:10:54.080] so basically we want to sell self dot grad to something and we want to set others dot grad

[01:10:59.680] to something okay and the way we saw below how chain rule works we want to take the local derivative

[01:11:07.840] times the sort of global derivative i should call it which is the derivative of the final output of

[01:11:17.120] the expression with respect to outs data with respect to out so the local derivative of self

[01:11:27.440] in an addition is 1.0 so it's just 1.0 times outs grad that's the chain rule and others dot grad

[01:11:37.520] 0.0 times grad and what you basically what you're seeing here is that outs grad will simply be

[01:11:43.840] copied onto selfs grad and others grad as we saw happens for an addition operation

[01:11:49.920] so we're going to later call this function to propagate the gradient having done an addition

[01:11:55.760] let's now do the multiplication we're going to also define that backward

[01:11:59.280] and we're going to set its backward to be backward

[01:12:07.840] and we want to chain out grad into self dot grad

[01:12:14.400] and others dot grad and this will be a little piece of chain rule for multiplication

[01:12:20.400] so we'll have so what should this be can you think through

[01:12:24.000] here so what is the local derivative here the local derivative was others dot data

[01:12:35.440] and then oops others dot data and then times out that grad that's channel

[01:12:42.560] and here we have self dot data times out that grad that's what we've been doing

[01:12:46.480] and finally here for 10h that backward and then we want to set outs backwards to be just backward

[01:12:57.520] and here we need to back propagate we have out that grad and we want to chain it into self dot grad

[01:13:06.480] and self dot grad will be the local derivative of this operation that we've done here which is 10h

[01:13:12.240] and so we saw that the local gradient is one minus the 10h of x squared which here is t

[01:13:19.840] that's the local derivative because that's t is the output of this 10h so one minus t square is

[01:13:24.720] the local derivative and then the gradient has to be multiplied because of the chain rule

[01:13:30.960] so out grad is chained through the local gradient into salt out grad and that should be basically

[01:13:36.880] it so we're going to redefine our value node we're going to swing all the way down here

[01:13:44.160] and we're going to redefine our expression make sure that all the grads are zero okay but now we

[01:13:52.960] don't have to do this manually anymore we are going to basically be calling the dot backward

[01:13:58.480] in the right order so first we want to call os dot backward so o was the outcome of 10h

[01:14:14.000] right so calling os that back those goes backward will be this function this is what it will do

[01:14:21.200] now we have to be careful because there's a times out dot grad and out dot grad remember is

[01:14:28.640] initialized to zero so here we see grad zero so as a base case we need to set os dot grad to 1.0

[01:14:41.920] to initialize this with one

[01:14:43.360] and then once this is one we can call o dot backward and what that should do is it should

[01:14:48.560] propagate this grad through 10h so the local derivative times the global derivative which

[01:14:55.520] is initialized at one so this should um

[01:15:01.760] uh don't so i thought about redoing it but i figured i should just leave the error in here

[01:15:10.720] because it's pretty funny why is an anti-object not callable uh it's because i screwed up we're

[01:15:18.080] trying to save these functions so this is correct so we're going to call this function

[01:15:23.600] this here you don't want to call the function because that returns none these functions return

[01:15:32.480] none we just want to store the function so let me redefine the value object and then we're going

[01:15:38.320] to come back in redefine the expression draw dot everything is great o dot grad is one

[01:15:44.720] o dot grad is one and now now this should work of course okay so all that backward should have

[01:15:53.280] this grad should now be 0.5 if we redraw and if everything went correctly 0.5 yay okay so now we

[01:16:01.200] need to call ends dot grad and it's not backward sorry ends backward and then we're going to call

[01:16:10.400] ends dot backward sorry ends backward so that seems to have worked

[01:16:17.840] so ends dot backward routed the gradient to both of these so this is looking great

[01:16:24.640] now we could of course call uh call b dot grad b dot backward sorry what's gonna happen

[01:16:32.000] well b doesn't have it backward b's backward because b is a leaf node b's backward is by

[01:16:38.960] initialization the empty function so nothing would happen but we can call call it on it

[01:16:45.840] but when we call this one it's backward

[01:16:53.520] then we expect this 0.5 to get further routed right so there we go 0.5 0.5

[01:16:59.840] and then finally we want to call it here on x2w2 and on x1w1

[01:17:14.720] let's do both of those and there we go so we get 0.5 negative 1.5 and 1 exactly as we did before

[01:17:24.160] but now we've done it through calling that backward sort of manually so we have one last

[01:17:32.880] piece to get rid of which is us calling underscore backward manually so let's think through what we

[01:17:38.400] are actually doing we've laid out a mathematical expression and now we're trying to go backwards

[01:17:43.840] through that expression so going backwards through the expression just means that we never want to

[01:17:49.600] call a dot backward for any node before we've done sort of everything after it so we have to do

[01:17:59.280] everything after it before we're ever going to call dot backward on any one node we have to get

[01:18:03.360] all of its full dependencies everything that it depends on has to propagate to it before we can

[01:18:08.960] continue back propagation so this ordering of graphs can be achieved using something called

[01:18:15.200] topological sort so topological sort is basically a laying out of a graph such that all the edges

[01:18:23.120] go only from left to right basically so here we have a graph it's a directed acyclic graph a DAG

[01:18:30.560] and this is two different topological orders of it I believe where basically you'll see that it's

[01:18:35.680] laying out of the nodes such that all the edges go only one way from left to right and implementing

[01:18:41.760] topological sort you can look in wikipedia and so on I'm not going to go through it in detail

[01:18:47.760] but basically this is what builds a topological graph we maintain a set of visited nodes and then

[01:18:56.160] we are going through starting at some root node which for us is O that's where I want to start

[01:19:02.960] the topological sort and starting at O we go through all of its children and we need to lay

[01:19:08.400] them out from left to right and basically this starts at O if it's not visited then it marks it

[01:19:16.080] as visited and then it iterates through all of its children and calls build topological on them

[01:19:23.120] and then after it's gone through all the children it adds itself so basically this node that we're

[01:19:30.080] going to call it on like say O is only going to add itself to the topo list after all of the

[01:19:36.400] children have been processed and that's how this function is guaranteeing that you're only not

[01:19:41.600] going to be in the list once all your children are in the list and that's the invariant that is

[01:19:45.600] being maintained so if we build topo on O and then inspect this list we're going to see that it

[01:19:52.320] ordered our value objects and the last one is the value of 0.707 which is the output so this is O

[01:20:01.760] and then this is N and then all the other nodes get laid out before it so that builds the topological

[01:20:10.080] graph and really what we're doing now is we're just calling dot underscore backward on all of

[01:20:16.000] the nodes in a topological order so if we just reset the gradients they're all zero what did we

[01:20:22.640] do we started by setting O dot grad to be one that's the base case then we built a topological

[01:20:34.320] order and then we went for node in reversed of topo now in in the reverse order because this

[01:20:48.080] list goes from you know we need to go through it in reversed order so starting at O node dot backward

[01:20:57.360] and this should be it there we go those are the correct derivatives finally we are going to hide

[01:21:07.600] this functionality so i'm going to copy this and we're going to hide it inside the value class

[01:21:13.520] because we don't want to have all that code lying around so instead of an underscore backward we're

[01:21:18.640] now going to define an actual backward so that backward without the underscore

[01:21:24.880] and that's going to do all the stuff that we just derived so let me just clean this up a little bit

[01:21:29.440] so we're first going to build a topological graph starting at self so build topo of self

[01:21:41.920] build topo of self will populate the topological order into the topo list which is a local variable

[01:21:49.280] then we set self dot grads to be one and then for each node in the reversed list

[01:21:55.920] so starting at s and going to all the children underscore backward and that should be it so

[01:22:04.800] save come down here we define okay all the grads are zero and now what we can do is

[01:22:13.360] O dot backward without the underscore and

[01:22:20.240] there we go and that's that's back propagation

[01:22:25.280] place for one neuron now we shouldn't be too happy with ourselves actually because we have

[01:22:30.400] a bad bug um and we have not surfaced the bug because of some specific conditions that we are

[01:22:36.080] have we have to think about right now so here's the simplest case that shows the bug say i create

[01:22:43.440] a single node a and then i create a b that is a plus a and then i call backward

[01:22:51.440] so what's going to happen is a is three and then a b is a plus a so there's two arrows on top of

[01:22:57.440] each other here then we can see that b is of course the forward pass works b is just a plus a

[01:23:05.200] which is six but the gradient here is not actually correct that we calculate it automatically and

[01:23:12.640] that we calculate it automatically and that's because um of course uh just doing calculus

[01:23:21.600] in your head the derivative of b with respect to a should be uh two one plus one it's not one

[01:23:30.800] intuitively what's happening here right so b is the result of a plus a

[01:23:34.400] and then we call backward on it so let's go up and see what that does

[01:23:39.120] um b is a result of addition so out is b and then when we call backward what happened is self.grad

[01:23:48.960] was set to one and then other.grad was set to one but because we're doing a plus a self and other

[01:23:58.240] are actually the exact same object so we are overriding the gradient we are setting it to one

[01:24:04.160] and then we are setting it again to one and that's why it stays at one so that's a problem

[01:24:11.760] there's another way to see this in a little bit more complicated expression

[01:24:18.720] so here we have a and b and then uh d will be the multiplication of the two and e will be the

[01:24:27.600] addition of the two and then we multiply e times d to get f and then we call it f.backward and

[01:24:35.200] these gradients if you check will be incorrect so fundamentally what's happening here again is

[01:24:42.400] basically we're going to see an issue anytime we use a variable more than once until now in these

[01:24:47.360] expressions above every variable is used exactly once so we didn't see the issue but here if a

[01:24:52.800] variable is used more than once what's going to happen during backward pass we're back propagating

[01:24:57.120] from f to e to d so far so good but now e calls it backward and it deposits its gradients to a and

[01:25:04.000] b but then we come back to d and call backward and it overwrites those gradients at a and b

[01:25:11.600] so that's obviously a problem and the solution here if you look at the multivariate case of the

[01:25:18.880] chain rule and its generalization there the solution there is basically that we have to

[01:25:23.680] accumulate these gradients these gradients add and so instead of setting those gradients

[01:25:31.760] we can simply do plus equals we need to accumulate those gradients plus equals plus equals plus

[01:25:39.120] equals plus equals plus equals plus equals and this will be okay remember because we are initializing

[01:25:49.120] them at zero so they start at zero and then any contribution that flows backwards will simply add

[01:25:58.800] so now if we redefine this one because the plus equals this now works because a dot grad started

[01:26:07.280] at zero and when we call b dot backward we deposit one and then we deposit one again and now this is

[01:26:13.360] two which is correct and here this will also work and we'll get correct gradients because when we

[01:26:18.960] call e dot backward we will deposit the gradients from this branch and then we get to back to d dot

[01:26:24.160] backward it will deposit its own gradients and then those gradients simply add on top of each

[01:26:29.360] other and so we just accumulate those gradients and that fixes the issue okay now before we move

[01:26:34.320] on let me actually do a bit of cleanup here and delete some of these some of this intermediate

[01:26:39.760] work so i'm not going to need any of this now that we've derived all of it um we are going to keep

[01:26:47.040] this because i want to come back to it delete the 10h delete harmonic example delete the step

[01:26:55.920] delete this keep the code that draws and then delete this example and leave behind only the

[01:27:03.280] definition of value and now let's come back to this non-linearity here that we implemented the

[01:27:08.320] 10h now i told you that we could have broken down 10h into its explicit atoms in terms of other

[01:27:15.360] expressions if we had the exp function so if you remember 10h is defined like this and we chose to

[01:27:21.040] develop 10h as a single function and we can do that because we know it's derivative and we can

[01:27:25.440] back propagate through it but we can also break down 10h into and express it as a function of exp

[01:27:31.200] and i would like to do that now because i want to prove to you that you get all the same results

[01:27:34.800] and all the same gradients but also because it forces us to implement a few more expressions

[01:27:39.840] it forces us to do exponentiation addition subtraction division and things like that and

[01:27:45.200] i think it's a good exercise to go through a few more of these okay so let's scroll up

[01:27:50.080] to the definition of value and here one thing that we currently can't do is we can do like a

[01:27:55.120] value of say 2.0 but we can't do you know here for example we want to add constant one and we

[01:28:02.720] can't do something like this and we can't do it because it says int object has no attribute data

[01:28:08.480] that's because a plus one comes right here to add and then other is the integer one and then here

[01:28:15.360] python is trying to access one dot data and that's not a thing that's because basically one is not a

[01:28:20.480] value object and we only have addition for value objects so as a matter of convenience so that we

[01:28:25.760] can create expressions like this and make them make sense we can simply do something like this

[01:28:32.320] basically we let other alone if other is an instance of value but if it's not an instance

[01:28:38.080] of value we're going to assume that it's a number like an integer or a float and we're going to

[01:28:41.520] simply wrap it in in value and then other will just become value of other and then other will

[01:28:46.720] have a data attribute and this should work so if i just save this redefine value then this should

[01:28:52.320] work there we go okay now let's do the exact same thing for multiply because we can't do something

[01:28:57.680] like this again for the exact same reason so we just have to go to mall and if other is not a

[01:29:05.280] value then let's wrap it in value let's redefine value and now this works now here's a kind of

[01:29:11.360] unfortunate and not obvious part a times two works we saw that but two times a is that going to work

[01:29:19.600] you'd expect it to write but actually it will not and the reason it won't is because python doesn't

[01:29:25.120] know like when you do a times two basically um so a times two python will go and it will basically

[01:29:32.400] do something like a dot mall of two that's basically what it will call but to it two times

[01:29:38.560] a is the same as two dot mall of a and it doesn't two can't multiply value and so it's really

[01:29:46.000] confused about that so instead what happens is in python the way this works is you are free to

[01:29:50.800] define something called the rmall and rmall is kind of like a fallback so if a python can't do

[01:29:58.800] two times a it will check if um if by any chance a knows how to multiply two and that will be called

[01:30:06.720] into rmall so because python can't do two times a it will check is there an rmall in value and

[01:30:13.680] because there is it will now call that and what we'll do here is we will swap the order of the

[01:30:19.520] operands so basically two times a will redirect to rmall and rmall will basically call a times two

[01:30:26.080] and that's how that will work so redefining that with rmall two times a becomes four okay now

[01:30:32.880] looking at the other elements that we still need we need to know how to exponentiate and how to divide

[01:30:37.040] so let's first the explanation do the exponentiation part we're going to introduce

[01:30:41.760] a single function exp here and exp is going to mirror 10h in the sense that it's a simple

[01:30:48.880] single function that transform a single scalar value and outputs a single scalar value

[01:30:53.120] so we pop out the python number we use math.exp to exponentiate it create a new value object

[01:30:58.640] everything that we've seen before the tricky part of course is how do you back propagate through

[01:31:02.720] e to the x and so here you can potentially pause the video and think about what should go here

[01:31:12.640] okay so basically we need to know what is the local derivative of e to the x so d by dx of e

[01:31:18.880] to the x is famously just e to the x and we've already just calculated e to the x and it's inside

[01:31:24.720] and it's inside out.data so we can do out.data times and out.grad that's the chain rule

[01:31:32.000] so we're just chaining on to the current running grad and this is what the expression looks like

[01:31:37.120] it looks a little confusing but this is what it is and that's the explanation

[01:31:41.760] so redefining we should now be able to call a.exp and hopefully the backward pass works as well

[01:31:48.160] okay and the last thing we'd like to do of course is we'd like to be able to divide

[01:31:51.440] now i actually will implement something slightly more powerful than division because division is

[01:31:56.080] just a special case of something a bit more powerful so in particular just by rearranging

[01:32:02.000] if we have some kind of a b equals value of 4.0 here we'd like to basically be able to do a divide

[01:32:08.000] b and we'd like this to be able to give us 0.5 now division actually can be reshuffled as follows

[01:32:14.720] if we have a divide b that's actually the same as a multiplying 1 over b and that's the same as a

[01:32:20.080] multiplying b to the power of negative 1 and so what i'd like to do instead is i'd basically like

[01:32:25.600] to implement the operation of x to the k for some constant k so it's an integer or a float

[01:32:33.280] and we would like to be able to differentiate this and then as a special case negative 1 will

[01:32:38.720] be division and so i'm doing that just because it's more general and yeah you might as well do

[01:32:44.880] yeah you might as well do it that way so basically what i'm saying is we can redefine uh division

[01:32:51.280] which we will put here somewhere yeah we can put it here somewhere what i'm saying is that we can

[01:32:57.200] redefine division so self-divide other this can actually be rewritten as self times other to the

[01:33:03.760] power of negative one and now a value raised to the power of negative one we have now defined that

[01:33:10.640] so here's so we need to implement the pow function where am i going to put the pow function maybe

[01:33:17.200] here somewhere there's this color for it so this function will be called when we try to raise a

[01:33:24.080] value to some power and other will be that power now i'd like to make sure that other is only an

[01:33:30.160] int or a float usually other is some kind of a different value object but here other will be

[01:33:35.600] forced to be an int or a float otherwise the math uh won't work for forward trying to achieve in

[01:33:42.320] the specific case that would be a different derivative expression if we wanted other to be

[01:33:47.120] a value so here we create the up the value which is just uh you know this data raised to the power

[01:33:53.040] of other and other here could be for example negative one that's what we are hoping to achieve

[01:33:58.320] and then uh this is the backward stub and this is the fun part which is what is the uh chain rule

[01:34:04.640] expression here for back for um back propagating through the power function where the power is to

[01:34:12.320] the power of some kind of a constant so this is the exercise and maybe pause the video here and

[01:34:16.800] see if you can figure it out yourself as to what we should put here okay so um you can actually go

[01:34:28.880] here and look at derivative rules as an example and we see lots of derivative rules that you can

[01:34:33.040] hopefully know from calculus in particular what we're looking for is the power rule

[01:34:37.920] because that's telling us that if we're trying to take d by dx of x to the n which is what we're

[01:34:42.000] doing here then that is just n times x to the n minus one right okay so that's telling us about

[01:34:51.520] the local derivative of this power operation so all we want here basically n is now other and

[01:34:59.840] self.data is x and so this now becomes other which is n times self.data which is now a python

[01:35:10.000] int or a float it's not a value object we're accessing the data attribute raised to the power

[01:35:16.800] of other minus one or n minus one i can put brackets around this but this doesn't matter

[01:35:22.080] because um power takes precedence over multiply in pyhelm so that would have been okay and then

[01:35:28.480] okay and that's the local derivative only but now we have to chain it and we change it just

[01:35:33.520] simply by multiplying by our top grad that's chain rule and this should uh technically work

[01:35:40.720] and we're gonna find out soon but now if we do this this should now work and we get 0.5

[01:35:47.760] so the forward pass works but does the backward pass work and i realized that we actually also

[01:35:52.480] have to know how to subtract so right now a minus b will not work to make it work we need one more

[01:36:00.080] piece of code here and basically this is the subtraction and the way we're going to implement

[01:36:07.120] subtraction is we're going to implement it by addition of a negation and then to implement

[01:36:11.520] negation we're going to multiply by negative one so just again using the stuff we've already built

[01:36:15.920] and just expressing it in terms of what we have and a minus b is not working okay so now let's

[01:36:21.760] scroll again to this expression here for this neuron and let's just uh compute the backward

[01:36:27.840] pass here once we've defined o and let's draw it so here's the gradients for all these leaf

[01:36:33.920] nodes for this two-dimensional neuron that has a 10h that we've seen before so now what i'd like

[01:36:39.040] to do is i'd like to break up this 10h into this expression here so let me copy paste this here

[01:36:46.560] and now instead of we'll preserve the label and we will change how we define o so in particular

[01:36:53.760] we're going to implement this formula here so we need e to the 2x minus 1 over e to the x plus 1

[01:36:59.840] so e to the 2x we need to take 2 times n and we need to exponentiate it that's e to the 2x

[01:37:07.040] and then because we're using it twice let's create an intermediate variable e and then define o as

[01:37:13.680] e plus 1 over e minus 1 over e plus 1 e minus 1 over e plus 1 and that should be it and then we

[01:37:23.520] should be able to draw dot of o so now before i run this what do we expect to see number one we're

[01:37:30.560] expecting to see a much longer graph here because we've broken up 10h into a bunch of other

[01:37:35.200] operations but those operations are mathematically equivalent and so what we're expecting to see is

[01:37:40.400] number one the same result here so the forward pass works and number two because of that

[01:37:46.000] mathematical equivalence we expect to see the same backward pass and the same gradients on these

[01:37:50.640] leaf nodes so these gradients should be identical so let's run this so number one let's verify that

[01:37:58.800] instead of a single 10h node we have now exp and we have plus we have times negative one this is

[01:38:06.160] the division and we end up with the same forward pass here and then the gradients we have to be

[01:38:11.760] careful because they're in slightly different order potentially the gradients for w2x2 should

[01:38:16.320] be 0 and 0.5 w2 and x2 are 0 and 0.5 and w1x1 are 1 and negative 1.5 1 and negative 1.5 so that means

[01:38:26.400] that both our forward passes and backward passes were correct because this turned out to be

[01:38:31.040] equivalent to 10h before and so the reason I wanted to go through this exercise is number one

[01:38:37.200] we got to practice a few more operations and writing more backwards passes and number two

[01:38:42.400] I wanted to illustrate the point that the level at which you implement your operations is totally

[01:38:49.040] up to you you can implement backward passes for tiny expressions like a single individual plus or

[01:38:53.760] a single times or you can implement them for say 10h which is a kind of a potentially you can see

[01:39:00.400] it as a composite operation because it's made up of all these more atomic operations but really all

[01:39:05.120] of this is kind of like a fake concept all that matters is we have some kind of inputs and some

[01:39:08.960] kind of an output and this output is a function of the inputs in some way and as long as you can

[01:39:12.960] do forward pass and the backward pass of that little operation it doesn't matter what that

[01:39:18.160] operation is and how composite it is if you can write the local gradients you can chain the

[01:39:23.680] gradient and you can continue back propagation so the design of what those functions are is

[01:39:28.400] completely up to you so now I would like to show you how you can do the exact same thing

[01:39:32.960] but using a modern deep neural network library like for example PyTorch which I've roughly modeled

[01:39:38.640] micrograd by and so PyTorch is something you would use in production and I'll show you how you can

[01:39:45.040] do the exact same thing but in PyTorch API so I'm just going to copy paste it in and walk you through

[01:39:50.160] it a little bit this is what it looks like so we're going to import PyTorch and then we need to

[01:39:55.600] define these value objects like we have here now micrograd is a scalar valued engine so we only have

[01:40:04.560] scalar values like 2.0 but in PyTorch everything is based around tensors and like I mentioned

[01:40:10.320] tensors are just n-dimensional arrays of scalars so that's why things get a little bit more

[01:40:15.840] complicated here I just need a scalar valued tensor a tensor with just a single element

[01:40:21.520] but by default when you work with PyTorch you would use more complicated tensors like this

[01:40:28.480] so if I import PyTorch then I can create tensors like this and this tensor for example is a 2x3

[01:40:36.880] array of scalars in a single compact representation so we can check its shape

[01:40:44.080] we see that it's a 2x3 array and so on so this is usually what you would work with in the actual

[01:40:50.640] libraries so here I'm creating a tensor that has only a single element 2.0 and then I'm casting it

[01:41:00.160] to be double because Python is by default using double precision for its floating point numbers so

[01:41:06.320] I'd like everything to be identical by default the data type of these tensors will be float 32

[01:41:12.640] so it's only using a single precision float so I'm casting it to double so that we have float 64

[01:41:18.880] just like in Python so I'm casting to double and then we get something similar to value of two

[01:41:25.760] the next thing I have to do is because these are leaf nodes by default PyTorch assumes that they

[01:41:29.840] do not require gradients so I need to explicitly say that all of these nodes require gradients

[01:41:35.920] so this is going to construct scalar valued one element tensors make sure that PyTorch knows that

[01:41:41.920] they require gradients now by default these are set to false by the way because of efficiency

[01:41:47.040] reasons because usually you would not want gradients for leaf nodes like the inputs to the

[01:41:52.640] network and this is just trying to be efficient in the most common cases so once we've defined

[01:41:57.760] all of our values in PyTorch land we can perform arithmetic just like we can here in micrograd land

[01:42:03.360] so this will just work and then there's a torch.tanh also and when we get back as a tensor

[01:42:08.960] again and we can just like in micrograd it's got a data attribute and it's got grad attributes

[01:42:15.840] so these tensor objects just like in micrograd have a dot data and a dot grad and we can

[01:42:21.920] run this in micrograd and the only difference here is that we need to call it dot item because

[01:42:28.400] otherwise PyTorch dot item basically takes a single tensor of one element and it just returns

[01:42:35.760] that element stripping out the tensor so let me just run this and hopefully we are going to get

[01:42:41.440] this is going to print the forward pass which is 0.707 and this will be the gradients which

[01:42:48.000] are 0.5 0 negative 1.5 and 1 so if we just run this there we go 0.7 so the forward pass agrees

[01:42:57.600] and then 0.5 0 negative 1.5 and 1 so PyTorch agrees with us and just to show you here basically

[01:43:04.480] oh here's a tensor with a single element and it's a double and we can call that item on it to just

[01:43:12.480] single number out so that's what item does and o is a tensor object like I mentioned and it's got

[01:43:19.360] a backward function just like we've implemented and then all of these also have a dot grad so

[01:43:24.480] like x2 for example has a grad and it's a tensor and we can pop out the individual number with dot

[01:43:29.680] item so basically Torch can do what we did in micrograd as a special case when your

[01:43:37.360] tensors are all single element tensors but the big deal with PyTorch is that everything is

[01:43:42.720] significantly more efficient because we are working with these tensor objects and we can do

[01:43:47.120] lots of operations in parallel on all of these tensors but otherwise what we've built very much

[01:43:53.600] agrees with the API of PyTorch okay so now that we have some machinery to build out pretty

[01:43:57.920] complicated mathematical expressions we can also start building up neural nets and as I mentioned

[01:44:02.560] neural nets are just a specific class of mathematical expressions so we're going to

[01:44:07.600] start building out a neural net piece by piece and eventually we'll build out a two-layer

[01:44:11.360] multi-layer layer perceptron as it's called and I'll show you exactly what that means let's start

[01:44:16.320] with a single individual neuron we've implemented one here but here I'm going to implement one that

[01:44:21.360] also subscribes to the PyTorch API in how it designs its neural network modules so just like

[01:44:27.840] we saw that we can like match the API of PyTorch on the autograd side we're going to try to do

[01:44:33.840] that on the neural network modules so here's class neuron and just for the sake of efficiency

[01:44:40.800] I'm going to copy paste some sections that are relatively straightforward

[01:44:45.520] so the constructor will take number of inputs to this neuron which is how many inputs come to a

[01:44:51.600] neuron so this one for example is three inputs and then it's going to create a weight that is some

[01:44:57.680] random number between negative one and one for every one of those inputs and a bias that controls

[01:45:02.960] the overall trigger happiness of this neuron and then we're going to implement a def underscore

[01:45:09.040] underscore call of self and x some input x and really what we don't want to do here is w times

[01:45:16.080] x plus b where w times x here is a dot product specifically now if you haven't seen call let me

[01:45:24.240] just return 0.0 here from now the way this works now is we can have an x which is say like 2.0 3.0

[01:45:30.720] then we can initialize a neuron that is two-dimensional because these are two

[01:45:34.640] numbers and then we can feed those two numbers into that neuron to get an output and so when

[01:45:40.320] you use this notation n of x python will use call so currently call just returns 0.0

[01:45:50.080] now we'd like to actually do the forward pass of this neuron instead so what we're going to

[01:45:55.440] do here first is we need to basically multiply all of the elements of w with all of the elements

[01:46:01.360] of x pairwise we need to multiply them so the first thing we're going to do is we're going to

[01:46:06.160] zip up salta w and x and in python zip takes two iterators and it creates a new iterator

[01:46:14.320] that iterates over the tuples of their corresponding entries

[01:46:17.920] so for example just to show you we can print this list and still return 0.0 here

[01:46:23.840] so we see that these w's are paired up with the x's w with x

[01:46:41.440] and now what we want to do is

[01:46:42.720] um for w i x i in we want to multiply w times w i times x i and then we want to sum all of that

[01:46:55.600] together to come up with an activation and add also salt b on top so that's the raw activation

[01:47:02.480] and then of course we need to pass that through a non-linearity so what we're going to be returning

[01:47:06.720] is act dot 10h and here's out so now we see that we are getting some outputs and we get a different

[01:47:15.120] output from a neuron each time because we are initializing different weights and biases and then

[01:47:20.480] to be a bit more efficient here actually sum by the way takes a second optional parameter which

[01:47:26.480] is the start and by default the start is zero so these elements of this sum will be added on top

[01:47:33.360] of zero to begin with but actually we can just start with cell dot b and then we just have an

[01:47:38.240] expression like this and then the generator expression here must be parenthesized in python

[01:47:52.400] yep so now we can forward a single neuron next up we're going to define a layer of neurons so here

[01:47:58.560] we have a schematic for a mlp so we see that these mlps each layer this is one layer has actually a

[01:48:05.760] number of neurons and they're not connected to each other but all of them are fully connected

[01:48:08.800] to the input so what is a layer of neurons it's just it's just a set of neurons evaluated

[01:48:13.760] independently so in the interest of time i'm going to do something fairly straightforward here

[01:48:20.400] it's literally a layer is just a list of neurons and then how many neurons do we have we take that

[01:48:28.560] as an input argument here how many neurons do you want in your layer number of outputs in this layer

[01:48:34.000] and so we just initialize completely independent neurons with this given dimensionality and we

[01:48:39.040] call on it we just independently evaluate them so now instead of a neuron we can make a layer

[01:48:46.160] of neurons they are two-dimensional neurons and let's have three of them and now we see that we

[01:48:50.480] have three independent evaluations of three different neurons right okay and finally let's

[01:48:57.200] complete this picture and define an entire multi-layer perceptron or mlp and as we can

[01:49:02.320] see here in an mlp these layers just feed into each other sequentially so let's come here and

[01:49:07.760] i'm just going to copy the code here in interest of time so an mlp is very similar to a multi-layer

[01:49:14.800] we're taking the number of inputs as before but now instead of taking a single nout which is

[01:49:19.920] number of neurons in a single layer we're going to take a list of nouts and this list defines the

[01:49:25.200] sizes of all the layers that we want in our mlp so here we just put them all together and then

[01:49:30.400] iterate over consecutive pairs of these sizes and create layer objects for them and then in the call

[01:49:36.720] function we are just calling them sequentially so that's an mlp really and let's actually

[01:49:41.600] re-implement this picture so we want three input neurons and then two layers of four and an output

[01:49:46.400] unit so we want a three-dimensional input say this is an example input we want three inputs

[01:49:54.880] into two layers of four and one output and this of course is an mlp and there we go that's a

[01:50:02.880] forward pass of an mlp to make this a little bit nicer you see how we have just a single element

[01:50:07.760] but it's wrapped in a list because layer always returns lists so for convenience returnouts at

[01:50:14.640] zero if lenouts is exactly a single element else returnfullest and this will allow us to just get

[01:50:21.520] a single value out at the last layer that only has a single neuron and finally we should be able to

[01:50:27.440] draw dot of n of x and as you might imagine these expressions are now getting relatively involved

[01:50:36.000] relatively involved so this is an entire mlp that we're defining now

[01:50:45.280] all the way until a single output okay and so obviously you would never differentiate

[01:50:51.600] on pen and paper these expressions but with micrograd we will be able to back propagate

[01:50:56.320] all the way through this and back propagate into these weights of all these neurons so let's see

[01:51:03.200] how that works okay so let's create ourselves a very simple example data set here so this data

[01:51:09.200] set has four examples and so we have four possible inputs into the neural net and we have four

[01:51:16.240] desired targets so we'd like the neural net to assign or output 1.0 when it's fed this example

[01:51:24.240] negative one when it's fed these examples and one when it's fed this example so it's a very simple

[01:51:28.560] binary classifier neural net basically that we would like here now let's think what the neural

[01:51:33.440] that currently thinks about these four examples we can just get their predictions basically we

[01:51:38.640] can just call n of x for x and x's and then we can print so these are the outputs of the neural net

[01:51:47.040] on those four examples so the first one is 0.91 but we'd like it to be one so we should push this

[01:51:55.120] one higher this one we want to be higher this one says 0.88 and we want this to be negative one

[01:52:02.640] this is 0.88 we want it to be negative one and this one is 0.88 we want it to be one

[01:52:08.240] so how do we make the neural net and how do we tune the weights to better predict the desired

[01:52:15.200] targets and the trick used in deep learning to achieve this is to calculate a single number that

[01:52:21.920] somehow measures the total performance of your neural net and we call this single number the loss

[01:52:28.000] so the loss first is a single number that we're going to define that basically measures how well

[01:52:34.560] the neural net is performing right now we have the intuitive sense that it's not performing very well

[01:52:38.480] because we're not very much close to this so the loss will be high and we'll want to minimize the

[01:52:43.760] loss so in particular in this case what we're going to do is we're going to implement the mean

[01:52:48.080] squared error loss so what this is doing is we're going to basically iterate for y ground truth

[01:52:56.400] and y output in zip of y's and y bread so we're going to pair up the ground truths with the

[01:53:04.800] predictions and the zip iterates over tuples of them and for each y ground truth and y output

[01:53:13.200] we're going to subtract them and square them so let's first see what these losses are these are

[01:53:20.560] individual loss components and so basically for each one of the four we are taking the

[01:53:27.440] prediction and the ground truth we are subtracting them and squaring them so because this one is so

[01:53:34.800] close to its target 0.91 is almost one subtracting them gives a very small number so here we would

[01:53:42.640] get like a negative 0.1 and then squaring it just makes sure that regardless of whether we

[01:53:49.280] are more negative or more positive we always get a positive number instead of squaring we should

[01:53:55.280] we could also take for example the absolute value we need to discard the sign and so you

[01:53:59.840] see that the expression is ranged so that you only get zero exactly when y out is equal to y

[01:54:05.600] ground truth when those two are equal so your prediction is exactly the target you are going

[01:54:09.760] to get zero and if your prediction is not the target you are going to get some other number

[01:54:15.040] so here for example we are way off and so that's why the loss is quite high and the more off we are

[01:54:21.920] the greater the loss will be so we don't want high loss we want low loss and so the final

[01:54:28.960] loss here will be just the sum of all of these numbers so you see that this should be zero

[01:54:36.320] roughly plus zero roughly but plus seven so loss should be about seven here and now we want to

[01:54:45.920] minimize the loss we want the loss to be low because if loss is low then every one of the

[01:54:52.960] predictions is equal to its target so the loss the lowest it can be is zero and the greater it is the

[01:55:00.480] worse off the neural net is predicting so now of course if we do loss that backward something magical

[01:55:08.640] happened when i hit enter and the magical thing of course that happened is that we can look at

[01:55:13.600] n dot layers dot neuron and dot layers at say like the first layer dot neurons at zero

[01:55:22.400] because remember that mlp has the layers which is a list and each layer has neurons which is a list

[01:55:28.720] and that gives us an individual neuron and then it's got some weights

[01:55:32.800] and so we can for example look at the weights at zero

[01:55:38.160] um oops it's not called weights it's called w and that's a value but now this value also has a grad

[01:55:48.000] because of the backward pass and so we see that because this gradient here on this particular

[01:55:54.240] weight of this particular neuron of this particular layer is negative we see that its influence on the

[01:55:59.360] loss is also negative so slightly increasing this particular weight of this neuron of this layer

[01:56:05.600] would make the loss go down and we actually have this information for every single one of our

[01:56:11.840] neurons and all their parameters actually it's worth looking at also a draw dot of loss by the

[01:56:16.880] way so previously we looked at the draw dot of a single neuron neuron forward pass and that was

[01:56:22.160] already a large expression but what is this expression we actually forwarded every one of

[01:56:27.920] those four examples and then we have the loss on top of them with the mean squared error and so

[01:56:33.120] this is a really massive graph because this graph that we've built up now oh my gosh this graph that

[01:56:41.280] we've built up now which is kind of excessive it's excessive because it has four forward passes of a

[01:56:46.960] neural net for every one of the examples and then it has the loss on top and it ends with the value

[01:56:52.640] of the loss which was 7.12 and this loss will now back propagate through all the forward passes

[01:56:58.720] all the way through just every single intermediate value of the neural net all the way back to of

[01:57:04.880] course the parameters of the weights which are the input so these weight parameters here are inputs

[01:57:10.880] to this neural net and these numbers here these scalars are inputs to the neural net so if we

[01:57:17.280] went around here we will probably find some of these examples this 1.0 potentially maybe this

[01:57:23.920] 1.0 or you know some of the others and you'll see that they all have gradients as well the thing is

[01:57:29.280] these gradients on the input data are not that useful to us and that's because the input data

[01:57:35.760] seems to be not changeable it's it's a given to the problem and so it's a fixed input we're not

[01:57:41.200] going to be changing it or messing with it even though we do have gradients for it but some of

[01:57:46.800] these gradients here will be for the neural network parameters the w's and the b's and those

[01:57:53.280] we of course we want to change okay so now we're going to want some convenience codes to gather up

[01:57:59.600] all the parameters of the neural net so that we can operate on all of them simultaneously and

[01:58:04.720] every one of them we will nudge a tiny amount based on the gradient information so let's collect the

[01:58:11.360] parameters of the neural net all in one array so let's create a parameters of self that just returns

[01:58:19.680] uh cell.w which is a list concatenated with a list of cell.b

[01:58:27.120] so this will just return a list list plus list just you know gives you a list so that's parameters

[01:58:33.200] of neuron and i'm calling it this way because also pytorch has parameters on every single

[01:58:38.800] and in module and it does exactly what we're doing here it just returns the parameter tensors

[01:58:45.360] for us it's the parameter scalars now layer is also a module so it will have parameters

[01:58:52.560] self and basically what we want to do here is something like this like

[01:58:57.680] uh params is here and then for neuron in self.neurons

[01:59:05.360] we want to get neuron.parameters and we want to params.extend

[01:59:12.000] right so these are the parameters of this neuron and then we want to put them on top of params so

[01:59:17.200] params.extend of piece and then we want to return params so this there's way too much code

[01:59:25.520] so actually there's a way to simplify this which is return p for neuron in self.neurons

[01:59:37.680] for p in neuron.parameters so it's a single list comprehension in python you can sort of nest them

[01:59:46.880] like this and you can then create the desired array so this is these are identical we can

[01:59:55.760] take this out and then let's do the same here def parameters self and return a parameter for layer

[02:00:09.040] in self.layers for p in layer.parameters and that should be good now let me pop out this

[02:00:23.280] so we don't reinitialize our network because we need to reinitialize our

[02:00:29.840] okay so unfortunately we will have to probably reinitialize the network because we just

[02:00:37.920] add functionality because this class of course we i want to get all the end up parameters but

[02:00:43.760] that's not going to work because this is the old class okay so unfortunately we do have to

[02:00:50.000] reinitialize the network which will change some of the numbers but let me do that so that we pick

[02:00:55.120] up the new api we can now do end up parameters and these are all the weights and biases inside

[02:01:01.440] the entire neural net so in total this mlp has 41 parameters and now we'll be able to change them

[02:01:13.680] if we recalculate the loss here we see that unfortunately we have slightly different

[02:01:18.160] predictions and slightly different loss but that's okay okay so we see that this neurons

[02:01:26.720] gradient is slightly negative we can also look at its data right now which is 0.85 so this is

[02:01:33.760] the current value of this neuron and this is its gradient on the loss so what we want to do now is

[02:01:40.400] we want to iterate for every p in end up parameters so for all the 41 parameters in this neural net

[02:01:50.960] we actually want to change p dot data slightly according to the gradient information okay so

[02:01:59.440] dot dot to do here but this will be basically a tiny update in this gradient descent scheme

[02:02:07.120] and in gradient descent we are thinking of the gradient as a vector pointing in the direction of

[02:02:14.240] increased loss and so in gradient descent we are modifying p dot data by a small step size

[02:02:24.800] in the direction of the gradient so the step size as an example could be like a very small number

[02:02:28.960] like 0.01 is the step size times p dot grad right but we have to think through some of the signs

[02:02:37.120] here so in particular working with this specific example here we see that if we just left it like

[02:02:44.880] this then this neuron's value would be currently increased by a tiny amount of the gradient

[02:02:52.160] the gradient is negative so this value of this neuron would go slightly down it would become

[02:02:57.280] like 0.8 you know 4 or something like that but if this neuron's value goes lower that would actually

[02:03:07.680] increase the loss that's because the derivative of this neuron is negative so increasing this makes

[02:03:16.320] the loss go down so increasing it is what we want to do instead of decreasing it so basically what

[02:03:22.320] we're missing here is we're actually missing a negative sign and again this other interpretation

[02:03:27.920] and that's because we want to minimize the loss we don't want to maximize the loss we want to

[02:03:31.280] decrease it and the other interpretation as i mentioned is you can think of the gradient

[02:03:35.360] vector so basically just the vector of all the gradients as pointing in the direction of increasing

[02:03:42.880] the loss but then we want to decrease it so we actually want to go in the opposite direction

[02:03:47.840] and so you can convince yourself that this sort of like that's the right thing here with this

[02:03:51.360] the right thing here with a negative because we want to minimize the loss

[02:03:55.440] so if we nudge all the parameters by a tiny amount

[02:04:00.640] then we'll see that this data will have changed a little bit so now this neuron is a tiny amount

[02:04:07.760] greater value so 0.854 went to 0.857 and that's a good thing because slightly increasing this neuron

[02:04:17.120] data makes the loss go down according to the gradient and so the correcting has happened

[02:04:22.640] sign-wise and so now what we would expect of course is that because we've changed all these

[02:04:28.480] parameters we expect that the loss should have gone down a bit so we want to re-evaluate the

[02:04:34.560] loss let me basically this is just a data definition that hasn't changed but the forward

[02:04:41.440] pass here of the network we can recalculate and actually let me do it outside here so that we can

[02:04:50.400] compare the two loss values so here if i recalculate the loss we'd expect the new loss

[02:04:57.200] now to be slightly lower than this number so hopefully what we're getting now is a tiny bit

[02:05:02.080] lower than 4.84 4.36 okay and remember the way we've arranged this is that low loss means that our

[02:05:11.200] predictions are matching the targets so our predictions now are probably slightly closer

[02:05:16.320] to the targets and now all we have to do is we have to iterate this process so again we've done

[02:05:23.840] the forward pass and this is the loss now we can lost that backward let me take these out and we

[02:05:30.080] can do a step size and now we should have a slightly lower loss 4.36 goes to 3.9 and okay so

[02:05:39.840] we've done the forward pass here's the backward pass nudge and now the loss is 3.66

[02:05:48.240] 3.47 and you get the idea we just continue doing this and this is gradient descent we're just

[02:05:55.600] iteratively doing forward pass backward pass update forward pass backward pass update and

[02:06:00.880] the neural net is improving its predictions so here if we look at ypred now ypred

[02:06:10.640] we see that this value should be getting closer to one so this value should be getting more

[02:06:15.840] positive these should be getting more negative and this one should be also getting more positive

[02:06:19.760] so if we just iterate this a few more times actually we may be able to afford to go a bit

[02:06:26.880] faster let's try a slightly higher learning rate oops okay there we go so now we're at 0.31

[02:06:37.360] if you go too fast by the way if you try to make it too big of a step you may actually overstep

[02:06:42.560] it's overconfidence because again remember we don't actually know exactly about the loss function

[02:06:46.720] the loss function has all kinds of structure and we only know about the very local dependence of

[02:06:51.840] all these parameters on the loss but if we step too far we may step into you know a part of the

[02:06:56.960] loss that is completely different and that can destabilize training and make your loss actually

[02:07:01.280] blow up even so the loss is now 0.04 so actually the predictions should be really

[02:07:08.000] quite close let's take a look so you see how this is almost one almost negative one almost one

[02:07:14.720] we can continue going so yep backward update oops there we go so we went way too fast and

[02:07:25.280] we actually overstepped so we got too too eager where are we now oops okay seven in negative

[02:07:34.000] nine so this is very very low loss and the predictions are basically perfect so somehow we

[02:07:43.120] basically we were doing way too big updates and we briefly exploded but then somehow we ended up

[02:07:47.200] getting into a really good spot so usually this learning rate and the tuning of it is a is a subtle

[02:07:52.960] art you want to set your learning rate if it's too low you're going to take way too long to converge

[02:07:58.000] but if it's too high the whole thing gets unstable and you're not going to be able to

[02:08:01.360] converge but if it's too high the whole thing gets unstable and you might actually even explode the

[02:08:06.000] loss depending on your loss function so finding the step size to be just right it's it's a pretty

[02:08:12.160] subtle art sometimes when you're using sort of vanilla gradient descent but we happen to get into

[02:08:16.800] a good spot we can look at n dot parameters so this is the setting of weights and biases that

[02:08:26.800] makes our network predict the desired targets very very close and basically we've successfully

[02:08:36.880] trained a neural net okay let's make this a tiny bit more respectable and implement an actual

[02:08:41.600] training loop and what that looks like so this is the data definition that stays this is the forward

[02:08:46.800] pass so for k in range you know we're going to take a bunch of steps first you do the forward

[02:08:58.880] pass we evaluate the loss let's reinitialize the neural net from scratch and here's the data

[02:09:06.960] and we first do forward pass then we do the backward pass and then we do an update that's

[02:09:19.360] gradient descent and then we should be able to iterate this and we should be able to print the

[02:09:27.600] current step the current loss let's just print the sort of number of the loss and then we should

[02:09:36.320] be good and that should be it and then the learning rate 0.01 is a little too small 0.1

[02:09:44.720] we saw is like a little bit dangerously too high let's go somewhere in between and we'll optimize

[02:09:50.080] this for not 10 steps but let's go for say 20 steps let me erase all of this junk

[02:09:57.680] and let's run the optimization and you see how we've actually converged slower

[02:10:04.720] in a more controlled manner and got to a loss that is very low so I expect widespread to be quite good

[02:10:14.480] there we go

[02:10:20.640] and that's it okay so this is kind of embarrassing but we actually have a really terrible bug

[02:10:26.240] in here and it's a subtle bug and it's a very common bug and I can't believe I've done it for

[02:10:31.920] the 20th time in my life especially on camera and I could have re-shot the whole thing but I think

[02:10:37.600] it's pretty funny and you know you get to appreciate a bit what working with neural nets

[02:10:42.880] maybe is like sometimes we are guilty of a common bug I've actually tweeted the most common neural

[02:10:51.840] that mistakes a long time ago now and I'm not really going to explain any of these except for

[02:10:59.520] we are guilty of number three you forgot to zero grad before dot backward what is that

[02:11:07.440] basically what's happening and it's a subtle bug and I'm not sure if you saw it is that all of these

[02:11:13.600] weights here have a dot data and a dot grad and dot grad starts at zero

[02:11:19.120] and then we do backward and we fill in the gradients and then we do an update on the data

[02:11:23.840] but we don't flush the grad it stays there so when we do the second forward pass and we do backward

[02:11:31.120] again remember that all the backward operations do a plus equals on the grad and so these gradients

[02:11:36.880] just add up and they never get reset to zero so basically we didn't zero grad so here's how we

[02:11:44.480] zero grad before backward we need to iterate over all the parameters and we need to make sure that

[02:11:51.840] p dot grad is set to zero we need to reset it to zero just like it is in the constructor so remember

[02:11:59.600] all the way here for all these value nodes grad is reset to zero and then all these backward passes

[02:12:05.360] do a plus equals from that grad but we need to make sure that we reset the values to zero

[02:12:11.760] so that when we do backward all of them start at zero and the actual backward pass accumulates

[02:12:18.240] the loss derivatives into the grads so this is zero grad in PyTorch and we will slightly

[02:12:27.360] we'll get a slightly different optimization let's reset the neural net the data is the same

[02:12:32.400] this is now I think correct and we get a much more you know we get a much more

[02:12:37.280] slower descent we still end up with pretty good results and we can continue this a bit more

[02:12:43.600] to get down lower and lower and lower yeah so the only reason that the previous thing worked

[02:12:52.960] it's extremely buggy the only reason that worked is that this is a very very simple problem

[02:13:00.000] and it's very easy for this neural net to fit this data and so the grads ended up accumulating

[02:13:06.480] and it effectively gave us a massive step size and it made us converge extremely fast

[02:13:13.760] but basically now we have to do more steps to get to very low values of loss and get

[02:13:19.520] Wipred to be really good we can try to step a bit greater and we can get a much better result

[02:13:25.920] we can try to step a bit greater yeah we're going to get closer and closer to one minus one and one

[02:13:38.400] so working with neural nets is sometimes tricky because

[02:13:44.960] you may have lots of bugs in the code and your network might actually work just like ours worked

[02:13:50.640] but chances are is that if we had a more complex problem then actually this bug would have made

[02:13:55.760] us not optimize the loss very well and we were only able to get away with it because

[02:14:00.880] the problem is very simple so let's now bring everything together and summarize what we learned

[02:14:06.080] what are neural nets neural nets are these mathematical expressions fairly simple

[02:14:11.120] mathematical expressions in the case of multilayered perceptron that take input as the

[02:14:16.400] input as the data and they take input the weights and the parameters of the neural net

[02:14:21.600] mathematical expression for the forward pass followed by a loss function and the loss function

[02:14:26.640] tries to measure the accuracy of the predictions and usually the loss will be low when your

[02:14:31.920] predictions are matching your targets or where the network is basically behaving well so we

[02:14:37.120] we manipulate the loss function so that when the loss is low the network is doing what you

[02:14:41.200] want it to do on your problem and then we backward the loss use back propagation to get the gradient

[02:14:48.800] and then we know how to tune all the parameters to decrease the loss locally

[02:14:52.800] but then we have to iterate that process many times in what's called the gradient descent

[02:14:56.800] so we simply follow the gradient information and that minimizes the loss and the loss is

[02:15:01.680] arranged so that when the loss is minimized the network is doing what you want it to do

[02:15:05.360] and yeah so we just have a blob of neural stuff and we can make it do arbitrary things

[02:15:11.600] and that's what gives neural nets their power it's you know this is a very tiny network with 41

[02:15:16.560] parameters but you can build significantly more complicated neural nets with billions

[02:15:23.040] at this point almost trillions of parameters and it's a massive blob of neural tissue simulated

[02:15:28.640] neural tissue roughly speaking and you can make it do extremely complex problems and these neural

[02:15:35.440] nets then have all kinds of very fascinating emergent properties in when you try to make them

[02:15:41.680] do significantly hard problems as in the case of gpt for example we have massive amounts of text

[02:15:48.640] from the internet and we're trying to get a neural nets to predict to take like a few words and try

[02:15:53.360] to predict the next word in a sequence that's the learning problem and it turns out that when you

[02:15:57.600] train this on all of internet the neural net actually has like really remarkable emergent

[02:16:01.920] properties but that neural net would have hundreds of billions of parameters but it works on

[02:16:07.440] fundamentally the exact same principles the neural net of course will be a bit more complex but

[02:16:12.400] otherwise the evaluating the gradient is there and would be identical and the gradient descent would

[02:16:19.200] be there and would be basically identical but people usually use slightly different updates

[02:16:23.760] this is a very simple stochastic gradient descent update and the loss function would not be a mean

[02:16:29.680] squared error they would be using something called the cross entropy loss for predicting the next

[02:16:34.000] token so there's a few more details but fundamentally the neural network setup and

[02:16:37.600] neural network training is identical and pervasive and now you understand intuitively how that works

[02:16:43.760] under the hood in the beginning of this video i told you that by the end of it you would understand

[02:16:47.760] everything in micrograd and then we'd slowly build it up let me briefly prove that to you

[02:16:52.080] so i'm going to step through all the code that is in micrograd as of today actually potentially

[02:16:56.720] some of the code will change by the time you watch this video because i intend to continue

[02:16:59.920] developing micrograd but let's look at what we have so far at least init.py is empty when you

[02:17:05.680] go to engine.py that has the value everything here you should mostly recognize so we have the

[02:17:10.720] data dot grad attributes we have the backward function we have the previous set of children

[02:17:15.520] and the operation that produced this value we have addition multiplication and raising to a

[02:17:21.760] scalar power we have the relu non-linearity which is slightly different type of non-linearity than

[02:17:26.720] tanh that we used in this video both of them are non-linearities and notably tanh is not actually

[02:17:32.160] present in micrograd as of right now but i intend to add it later we have the backward which is

[02:17:37.840] identical and then all of these other operations which are built up on top of operations here

[02:17:43.040] so value should be very recognizable except for the non-linearity used in this video

[02:17:48.400] there's no massive difference between relu and tanh and sigmoid and these other non-linearities

[02:17:52.720] they're all roughly equivalent and can be used in mlps so i use tanh because it's a bit smoother

[02:17:57.760] and because it's a little bit more complicated than relu and therefore it's stressed a little

[02:18:02.000] bit more the local gradients and working with those derivatives which i thought would be useful

[02:18:06.960] nn.py is the neural networks library as i mentioned so you should recognize identical

[02:18:11.600] implementation of neuron layer and mlp notably or not so much we have a class module here there's a

[02:18:18.160] parent class of all these modules i did that because there's an n.module class in pytorch

[02:18:23.920] and so this exactly matches that api and n.module in pytorch has also a zero grad which i refactored

[02:18:29.680] out here so that's the end of micrograd really then there's a test which you'll see basically

[02:18:38.000] creates two chunks of code one in micrograd and one in pytorch and we'll make sure that the forward

[02:18:43.920] and the backward paths agree identically for a slightly less complicated expression a slightly

[02:18:48.560] more complicated expression everything agrees so we agree with pytorch on all of these operations

[02:18:54.080] and finally there's a demo that i.py y and b here and it's a bit more complicated binary classification

[02:18:58.800] demo than the one i covered in this lecture so we only had a tiny data set of four examples

[02:19:04.480] here we have a bit more complicated example with lots of blue points and lots of red points

[02:19:09.200] and we're trying to again build a binary classifier to distinguish two-dimensional

[02:19:13.280] points as red or blue it's a bit more complicated mlp here with it's a bigger

[02:19:18.560] mlp the loss is a bit more complicated because it supports batches so because our data set was

[02:19:25.360] so tiny we always did a forward pass on the entire data set of four examples but when your data set

[02:19:30.480] is like a million examples what we usually do in practice is we basically pick out some random

[02:19:36.240] subset we call that a batch and then we only process the batch forward backward and update

[02:19:41.360] so we don't have to forward the entire training set and then we have a batch of four examples

[02:19:46.400] so we don't have to forward the entire training set so this supports batching because there's

[02:19:51.360] a lot more examples here we do a forward pass the loss is slightly more different this is a

[02:19:57.280] max margin loss that i implement here the one that we used was the mean squared error loss

[02:20:02.640] because it's the simplest one there's also the binary cross entropy loss all of them can be used

[02:20:07.600] for binary classification and don't make too much of a difference in the simple examples that we

[02:20:11.680] looked at so far there's something called l2 regularization used here this has to do with

[02:20:17.840] generalization of the neural net and controls the overfitting in machine learning setting

[02:20:22.480] but i did not cover these concepts in this video potentially later and the training

[02:20:27.360] loop you should recognize so forward backward with zero grad and update and so on you'll

[02:20:34.720] notice that in the update here the learning rate is scaled as a function of number of iterations

[02:20:39.440] and it shrinks and this is something called learning rate decay so in the beginning you

[02:20:44.640] have a high learning rate and as the network sort of stabilizes near the end you bring down the

[02:20:49.840] learning rate to get some of the fine details in the end and in the end we see the decision

[02:20:54.800] surface of the neural net and we see that it learned to separate out the red and the blue

[02:20:58.960] area based on the data points so that's the slightly more complicated example in the demo

[02:21:04.560] demo.hypyymb that you're free to go over but yeah as of today that is micrograd i also wanted

[02:21:10.640] to show you a little bit of real stuff so that you get to see how this is actually implemented

[02:21:14.160] in a production grade library like pytorch so in particular i wanted to show i wanted to find and

[02:21:19.360] show you the backward pass for 10h in pytorch so here in micrograd we see that the backward

[02:21:25.120] password 10h is 1 minus t square where t is the output of the 10h of x times out that grad which

[02:21:34.480] is the chain rule so we're looking for something that looks like this now i went to pytorch which

[02:21:41.520] has an open source github code base and i looked through a lot of its code and honestly i spent

[02:21:49.040] about 15 minutes and i couldn't find 10h and that's because these libraries unfortunately

[02:21:53.920] they grow in size and entropy and if you just search for 10h you get apparently 2800 results

[02:22:00.320] and 400 and 406 files so i don't know what these files are doing honestly and why there are so many

[02:22:08.560] mentions of 10h but unfortunately these libraries are quite complex they're meant to be used not

[02:22:13.600] really inspected eventually i did stumble on someone who tries to change the 10h backward

[02:22:21.120] code for some reason and someone here pointed to the cpu kernel and the cuda kernel for 10h backward

[02:22:27.760] so this so basically depends on if you're using pytorch on a cpu device or on a gpu

[02:22:32.320] which these are different devices and i haven't covered this but this is the 10h backward kernel

[02:22:37.760] for cpu and the reason it's so large is that number one this is like if you're using a complex

[02:22:45.920] type which we haven't even talked about if you're using a specific data type of bfloat 16 which we

[02:22:50.480] haven't talked about and then if you're not then this is the kernel and deep here we see something

[02:22:57.360] that resembles our backward pass so they have a times one minus b square so this b here must be

[02:23:06.560] the output of the 10h and this is the out.grad so here we found it deep inside pytorch on this

[02:23:14.800] location for some reason inside binary ops kernel when 10h is not actually a binary op and then this

[02:23:21.760] is the gpu kernel we're not complex we're here and here we go with one line of code so we did

[02:23:31.600] find it but basically unfortunately these code bases are very large and micrograd is very very

[02:23:37.760] simple but if you actually want to use real stuff finding the code for it you'll actually find that

[02:23:42.240] difficult i also wanted to show you a little example here where pytorch is showing you how

[02:23:48.240] you can register a new type of function that you want to add to pytorch as a lego building block

[02:23:53.600] so here if you want to for example add a genre polynomial three here's how you could do it you

[02:24:00.480] will register it as a class that subclasses torch.rgrad.function and then you have to tell

[02:24:07.040] pytorch how to forward your new function and how to backward through it so as long as you

[02:24:13.360] can do the forward pass of this little function piece that you want to add and as long as you

[02:24:16.960] know the the local derivative the local gradients which are implemented in the backward pytorch

[02:24:21.920] will be able to back propagate through your function and then you can use this as a lego block

[02:24:25.920] in a larger lego castle of all the different lego blocks that pytorch already has and so that's the

[02:24:31.760] only thing you have to tell pytorch and everything would just work and you can register new types of

[02:24:35.760] functions in this way following this example and that is everything that i wanted to cover in this

[02:24:40.720] lecture so i hope you enjoyed building out micrograd with me i hope you find it interesting

[02:24:45.120] insightful and yeah i will post a lot of the links that are related to this video in the video

[02:24:51.920] description below i will also probably post a link to a discussion forum or discussion group where

[02:24:57.440] you can ask questions related to this video and then i can answer or someone else can answer your

[02:25:02.480] questions and i may also do a follow-up video that answers some of the most common questions

[02:25:08.400] but for now that's it i hope you enjoyed it if you did then please like and subscribe

[02:25:12.480] so that youtube knows to feature this video to more people and that's it for now i'll see you later

[02:25:22.400] now here's the problem we know dl by wait what is the problem

[02:25:29.520] and that's everything i wanted to cover in this lecture so i hope um you enjoyed us building out

[02:25:34.640] micrograb micrograb okay now let's do the exact same thing for multiply because we can't do

[02:25:42.240] something like a times two oops i know what happened there

