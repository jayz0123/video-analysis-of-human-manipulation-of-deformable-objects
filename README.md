# video-analysis-of-human-manipulation-of-deformable-objects
MSc project at Imperial College London

## Project Discription
There is considerable work on recognising and learning actions from videos, with applications to automatic caption generation,
as well as learning robot control from demonstrations (e.g. [1]. Yet many of these actions have been restricted either to
gestural/facial movements, or actions on rigid objects, with work on deformable objects (e.g. clothes, or strings) to be
considerably less [see reviews [2,3] below).
In this project you will use computer vision to analyse a video, recognise and track the state of a deformable object, as well as
human hand actions, and output a sequence of object-oriented actions, for further processing in potential extensions (for
example, generating a textual or speech description of what happened, or controlling a bimanual robot to repeat the
demonstrated action, for those students interested in robot applications). Since the general problem is very challenging, we will
constraint it to tracking and recognising operations on deformable strings, for example shoe laces or computer cables. For
example, your processing pipeline will take a video of a human tying a shoe, or plugging/unplugging the cables on the back of a
server cabinet) and output an analysis of the base object (e.g. for the shoe this is the locations of the holes) and a task solution
representation (a sequence of actions that were performed, for example using (but not restricted to if student wants to use
something else) to action grammars [1].
This is a challenging project, with several computer vision challenges. You should have strong programming/algorithmic
background (e.g. Python/C++) since you will be programming a fairly elaborate computer vision processing pipeline. Strong
interest in AI is preferred, since you will be researching algorithms and data structures for representing both human movements
and the deformation of strings.

- [1] K Lee et al, A syntactic approach to robot imitation learning using probabilistic activity grammars, Robotics and Autonomous
Systems Journal, http://khlee.org/papers/lee-ras13.pdf
- [2] Hu et al, A review on modeling of flexible deformable object for dexterous robotic manipulation, IJARS,
https://journals.sagepub.com/doi/full/10.1177/1729881419848894
- [3] Sanchez et al, Robotic manipulation and sensing of deformable objects in domestic and industrial applications: a survey,
IJRR, 2018, https://journals.sagepub.com/doi/pdf/10.1177/0278364918779698

## Contents
### [`Docs`](docs)
Ducuments which includes important information as well as some ideas regarding this project.
