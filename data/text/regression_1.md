<div style="text-align: justify;">
In conformal prediction for regression, we aim to construct prediction intervals around our point estimates to offer a statistically 
valid level of confidence. The method leverages a calibration dataset to compute 'conformity scores,' which help us rank 
how well the model's predictions align with actual outcomes. These scores, in turn, guide the creation of prediction intervals 
with a desired coverage probability. Thus, conformal prediction serves as a tool for robust and interpretable prediction intervals.
</div>

<br>

<div style="text-align: justify;">We now turn our attention to an example first presented by Neil Lawrence in his article, <a href='https://inverseprobability.com/talks/notes/deep-gaussian-processes.html'>Deep Gaussian Processes I</a>. Alan Turing, widely regarded as the father of modern computing, was also an exceptional athlete. He completed a marathon&mdash;spanning a distance of 26 miles and 385 yards&mdash;in a mere 2 hours, 46 minutes, and 3 seconds<sup><a href='#references'>{}</a></sup>. This achievement translates to an impressive pace of approximately 3.94 minutes per kilometer. By employing conformal prediction, we aim to estimate whether Turing would have secured a hypothetical Olympic gold medal in the marathon, had the Olympics been staged in 1946. Let us explore the possibility of his winning the medal.</div>
