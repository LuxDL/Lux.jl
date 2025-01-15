import{_ as s,c as a,o as n,a3 as i}from"./chunks/framework.Ch6DX-It.js";const y=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),p={name:"tutorials/beginner/3_SimpleRNN.md"},l=i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxAMDGPU, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">      Statistics</span></span></code></pre></div><h2 id="Dataset" tabindex="-1">Dataset <a class="header-anchor" href="#Dataset" aria-label="Permalink to &quot;Dataset {#Dataset}&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create the spirals</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [MLUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Datasets</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">make_spiral</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(sequence_length) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> _ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">dataset_size]</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Get the labels</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    labels </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vcat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">repeat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">([</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">repeat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">([</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    clockwise_spirals </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(d[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">][:, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">sequence_length], :, sequence_length, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">                         for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> d </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)]]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    anticlockwise_spirals </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                                 d[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">][:, (sequence_length </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], :, sequence_length, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">                             for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> d </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data[((dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Float32</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">cat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(clockwise_spirals</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, anticlockwise_spirals</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; dims</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Split the dataset</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (x_train, y_train), (x_val, y_val) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> splitobs</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x_data, labels); at</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create DataLoaders</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Use DataLoader to automatically minibatch and shuffle the data</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">collect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.((x_train, y_train)); batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Don&#39;t shuffle the validation data</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">collect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.((x_val, y_val)); batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractExplicitContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/v0.5.49/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">:</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">       Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">AbstractExplicitContainerLayer{(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/v0.5.49/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/v0.5.49/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>Main.var&quot;##225&quot;.SpiralClassifier</span></span></code></pre></div><p>We can use default Lux blocks – <code>Recurrence(LSTMCell(in_dims =&gt; hidden_dims)</code> – instead of defining the following. But let&#39;s still do it for the sake of it.</p><p>Now we need to define the behavior of the Classifier when it is invoked.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray{T, 3}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedTuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedTuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {T}</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # First we will have to run the sequence through the LSTM Cell</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # The first call to LSTM Cell will create the initial hidden state</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # See that the parameters and states are automatically populated into a field called</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # \`lstm_cell\` We use \`eachslice\` to get the elements in the sequence without copying,</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # and \`Iterators.peel\` to split out the first element for LSTM initialization.</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_init, x_rest </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Iterators</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">peel</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">_eachslice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (y, carry), st_lstm </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_init, ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lstm_cell, st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lstm_cell)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Now that we have the hidden state and memory in \`carry\` we will pass the input and</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # \`carry\` jointly</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x_rest</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        (y, carry), st_lstm </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x, carry), ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lstm_cell, st_lstm)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # After running through the sequence we will pass the output through the classifier</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y, st_classifier </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y, ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">classifier, st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">classifier)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Finally remember to create the updated state</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> merge</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st, (classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">st_classifier, lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">st_lstm))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vec</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y), st</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/v0.5.49/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; lstm_cell, classifier) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray{T, 3}</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {T}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        x_init, x_rest </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Iterators</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">peel</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">_eachslice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        y, carry </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_init)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x_rest</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            y, carry </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x, carry))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vec</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>SpiralClassifierCompact (generic function with 1 method)</span></span></code></pre></div><h2 id="Defining-Accuracy,-Loss-and-Optimiser" tabindex="-1">Defining Accuracy, Loss and Optimiser <a class="header-anchor" href="#Defining-Accuracy,-Loss-and-Optimiser" aria-label="Permalink to &quot;Defining Accuracy, Loss and Optimiser {#Defining-Accuracy,-Loss-and-Optimiser}&quot;">​</a></h2><p>Now let&#39;s define the binarycrossentropy loss. Typically it is recommended to use <code>logitbinarycrossentropy</code> since it is more numerically stable, but for the sake of simplicity we will use <code>binarycrossentropy</code>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> xlogy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, y)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    result </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> log</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> ifelse</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">iszero</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">zero</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(result), result)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> binarycrossentropy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y_pred </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y_pred </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> eps</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">eltype</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> mean</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@.</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> -</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">xlogy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_true, y_pred) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> xlogy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> -</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y_true, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> -</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y_pred))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> compute_loss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, (x, y))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y_pred, st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, ps, st)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> binarycrossentropy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y), st, (; y_pred</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">y_pred)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">matches</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> sum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((y_pred </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.5f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.==</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y_true)</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> matches</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>accuracy (generic function with 1 method)</span></span></code></pre></div><h2 id="Training-the-Model" tabindex="-1">Training the Model <a class="header-anchor" href="#Training-the-Model" aria-label="Permalink to &quot;Training the Model {#Training-the-Model}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model_type)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Get the dataloaders</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (train_loader, val_loader) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model_type</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    rng </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Xoshiro</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    dev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> gpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Experimental</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TrainState</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        rng, model, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Adam</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.01f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">); transform_variables</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">dev)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">25</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Train the model</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> train_loader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            gs, loss, _, train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Experimental</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">compute_gradients</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                AutoZygote</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), compute_loss, (x, y), train_state)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Experimental</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">apply_gradients!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state, gs)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Epoch [%3d]: Loss %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch loss</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Validate the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> val_loader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            loss, st_, ret </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> compute_loss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, st_, (x, y))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ret</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">y_pred, y)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Validation: Loss %4.5f Accuracy %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> loss acc</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> cpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>Epoch [  1]: Loss 0.56230</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50787</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47183</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.44831</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.42515</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.39968</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.38847</span></span>
<span class="line"><span>Validation: Loss 0.37128 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36232 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36567</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.35083</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.32326</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.31313</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.29613</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.29390</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.26999</span></span>
<span class="line"><span>Validation: Loss 0.25869 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.25399 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.25840</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23970</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22939</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22172</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.20551</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19663</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19071</span></span>
<span class="line"><span>Validation: Loss 0.17909 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.17725 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17683</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16940</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16207</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15299</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14434</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13954</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13248</span></span>
<span class="line"><span>Validation: Loss 0.12694 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12591 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12561</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12243</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11473</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10923</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10484</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09969</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09117</span></span>
<span class="line"><span>Validation: Loss 0.09210 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09087 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09119</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08721</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08429</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08053</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07627</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07256</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.06711</span></span>
<span class="line"><span>Validation: Loss 0.06816 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06653 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06757</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06418</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06203</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05897</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05659</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05474</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05000</span></span>
<span class="line"><span>Validation: Loss 0.05117 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04938 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05164</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04705</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04762</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04371</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04206</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04097</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.03638</span></span>
<span class="line"><span>Validation: Loss 0.03881 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03704 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03905</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03536</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03580</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03409</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03250</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.02967</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.02998</span></span>
<span class="line"><span>Validation: Loss 0.02991 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02827 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02873</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02851</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02644</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02676</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02606</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02348</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02260</span></span>
<span class="line"><span>Validation: Loss 0.02368 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02222 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02268</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02264</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02253</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02078</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01990</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01909</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01871</span></span>
<span class="line"><span>Validation: Loss 0.01934 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01808 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01856</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01859</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01782</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01613</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01744</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01647</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01616</span></span>
<span class="line"><span>Validation: Loss 0.01628 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01520 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01541</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01563</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01470</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01589</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01363</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01424</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01273</span></span>
<span class="line"><span>Validation: Loss 0.01406 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01313 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01424</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01366</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01299</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01243</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01298</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01158</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01155</span></span>
<span class="line"><span>Validation: Loss 0.01240 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01158 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01231</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01144</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01168</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01154</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01102</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01106</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01025</span></span>
<span class="line"><span>Validation: Loss 0.01110 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01037 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01064</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01049</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01047</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00999</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01023</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01026</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00929</span></span>
<span class="line"><span>Validation: Loss 0.01006 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00939 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01024</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00965</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00946</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00929</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00959</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00816</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00874</span></span>
<span class="line"><span>Validation: Loss 0.00919 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00857 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00881</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00875</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00920</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00820</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00820</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00842</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00803</span></span>
<span class="line"><span>Validation: Loss 0.00845 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00788 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00827</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00772</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00804</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00804</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00791</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00741</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00786</span></span>
<span class="line"><span>Validation: Loss 0.00781 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00728 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00706</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00747</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00737</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00740</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00702</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00752</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00742</span></span>
<span class="line"><span>Validation: Loss 0.00726 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00676 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00680</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00692</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00706</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00728</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00658</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00612</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00699</span></span>
<span class="line"><span>Validation: Loss 0.00676 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00630 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00669</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00615</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00620</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00679</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00600</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00630</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00616</span></span>
<span class="line"><span>Validation: Loss 0.00633 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00589 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00643</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00583</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00599</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00578</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00626</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00562</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00493</span></span>
<span class="line"><span>Validation: Loss 0.00594 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00553 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00601</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00548</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00526</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00567</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00569</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00551</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00508</span></span>
<span class="line"><span>Validation: Loss 0.00559 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00520 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00539</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00515</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00546</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00503</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00528</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00533</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00500</span></span>
<span class="line"><span>Validation: Loss 0.00528 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00491 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>Epoch [  1]: Loss 0.56248</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51458</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48072</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.45577</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.43041</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.40489</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.38433</span></span>
<span class="line"><span>Validation: Loss 0.36590 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36632 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36544</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.35846</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.33377</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.31587</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.30113</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.29569</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28527</span></span>
<span class="line"><span>Validation: Loss 0.25747 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.25652 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.26236</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24692</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23333</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22068</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21304</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.20160</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19337</span></span>
<span class="line"><span>Validation: Loss 0.18109 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.17993 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.18231</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17341</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16459</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15686</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15184</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14197</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13611</span></span>
<span class="line"><span>Validation: Loss 0.12983 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12897 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12916</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12544</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11855</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11328</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10950</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10356</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09961</span></span>
<span class="line"><span>Validation: Loss 0.09428 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09376 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09507</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09080</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08619</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08269</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07970</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07736</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07522</span></span>
<span class="line"><span>Validation: Loss 0.06923 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06901 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06988</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06733</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06386</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06356</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05810</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05724</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05489</span></span>
<span class="line"><span>Validation: Loss 0.05140 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05138 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05414</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04993</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04671</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04568</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04472</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04278</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04363</span></span>
<span class="line"><span>Validation: Loss 0.03847 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03856 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03891</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03931</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03689</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03574</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03315</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03136</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03059</span></span>
<span class="line"><span>Validation: Loss 0.02921 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02934 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03010</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02901</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02917</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02671</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02502</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02494</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02767</span></span>
<span class="line"><span>Validation: Loss 0.02283 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02298 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02308</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02403</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02281</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02141</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02036</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01984</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02063</span></span>
<span class="line"><span>Validation: Loss 0.01850 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01863 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01859</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01850</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01799</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01830</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01793</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01714</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01552</span></span>
<span class="line"><span>Validation: Loss 0.01551 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01561 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01598</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01515</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01463</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01499</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01540</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01569</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01371</span></span>
<span class="line"><span>Validation: Loss 0.01337 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01346 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01438</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01338</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01328</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01347</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01312</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01245</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01167</span></span>
<span class="line"><span>Validation: Loss 0.01178 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01185 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01268</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01226</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01078</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01249</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01085</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01165</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01146</span></span>
<span class="line"><span>Validation: Loss 0.01053 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01060 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01183</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01098</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01024</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01003</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01042</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01055</span></span>
<span class="line"><span>Validation: Loss 0.00952 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00958 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00961</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00990</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00988</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00918</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00968</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00932</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00968</span></span>
<span class="line"><span>Validation: Loss 0.00868 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00873 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00899</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00831</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00917</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00907</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00836</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00884</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00855</span></span>
<span class="line"><span>Validation: Loss 0.00797 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00802 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00837</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00777</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00851</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00779</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00817</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00805</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00749</span></span>
<span class="line"><span>Validation: Loss 0.00736 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00740 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00780</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00755</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00798</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00762</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00675</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00723</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00741</span></span>
<span class="line"><span>Validation: Loss 0.00683 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00687 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00709</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00691</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00693</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00683</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00725</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00669</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00710</span></span>
<span class="line"><span>Validation: Loss 0.00636 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00640 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00656</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00672</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00692</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00585</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00658</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00623</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00684</span></span>
<span class="line"><span>Validation: Loss 0.00595 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00598 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00560</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00635</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00610</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00596</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00632</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00623</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00568</span></span>
<span class="line"><span>Validation: Loss 0.00558 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00561 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00576</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00603</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00541</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00561</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00591</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00551</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00582</span></span>
<span class="line"><span>Validation: Loss 0.00525 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00528 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00542</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00535</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00489</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00584</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00537</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00534</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00552</span></span>
<span class="line"><span>Validation: Loss 0.00495 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00498 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {compress </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">} ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
<span class="line"><span> :ps_trained</span></span>
<span class="line"><span> :st_trained</span></span></code></pre></div><h2 id="Appendix" tabindex="-1">Appendix <a class="header-anchor" href="#Appendix" aria-label="Permalink to &quot;Appendix {#Appendix}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">InteractiveUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxCUDA) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&amp;&amp;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> CUDA</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    CUDA</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxAMDGPU) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&amp;&amp;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> LuxAMDGPU</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    AMDGPU</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>Julia Version 1.10.3</span></span>
<span class="line"><span>Commit 0b4590a5507 (2024-04-30 10:59 UTC)</span></span>
<span class="line"><span>Build Info:</span></span>
<span class="line"><span>  Official https://julialang.org/ release</span></span>
<span class="line"><span>Platform Info:</span></span>
<span class="line"><span>  OS: Linux (x86_64-linux-gnu)</span></span>
<span class="line"><span>  CPU: 48 × AMD EPYC 7402 24-Core Processor</span></span>
<span class="line"><span>  WORD_SIZE: 64</span></span>
<span class="line"><span>  LIBM: libopenlibm</span></span>
<span class="line"><span>  LLVM: libLLVM-15.0.7 (ORCJIT, znver2)</span></span>
<span class="line"><span>Threads: 8 default, 0 interactive, 4 GC (on 2 virtual cores)</span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>  JULIA_CPU_THREADS = 2</span></span>
<span class="line"><span>  JULIA_AMDGPU_LOGGING_ENABLED = true</span></span>
<span class="line"><span>  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6</span></span>
<span class="line"><span>  LD_LIBRARY_PATH = /usr/local/nvidia/lib:/usr/local/nvidia/lib64</span></span>
<span class="line"><span>  JULIA_PKG_SERVER = </span></span>
<span class="line"><span>  JULIA_NUM_THREADS = 8</span></span>
<span class="line"><span>  JULIA_CUDA_HARD_MEMORY_LIMIT = 100%</span></span>
<span class="line"><span>  JULIA_PKG_PRECOMPILE_AUTO = 0</span></span>
<span class="line"><span>  JULIA_DEBUG = Literate</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA runtime 12.4, artifact installation</span></span>
<span class="line"><span>CUDA driver 12.4</span></span>
<span class="line"><span>NVIDIA driver 550.54.15</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA libraries: </span></span>
<span class="line"><span>- CUBLAS: 12.4.5</span></span>
<span class="line"><span>- CURAND: 10.3.5</span></span>
<span class="line"><span>- CUFFT: 11.2.1</span></span>
<span class="line"><span>- CUSOLVER: 11.6.1</span></span>
<span class="line"><span>- CUSPARSE: 12.3.1</span></span>
<span class="line"><span>- CUPTI: 22.0.0</span></span>
<span class="line"><span>- NVML: 12.0.0+550.54.15</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Julia packages: </span></span>
<span class="line"><span>- CUDA: 5.3.4</span></span>
<span class="line"><span>- CUDA_Driver_jll: 0.8.1+0</span></span>
<span class="line"><span>- CUDA_Runtime_jll: 0.12.1+0</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Toolchain:</span></span>
<span class="line"><span>- Julia: 1.10.3</span></span>
<span class="line"><span>- LLVM: 15.0.7</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>- JULIA_CUDA_HARD_MEMORY_LIMIT: 100%</span></span>
<span class="line"><span></span></span>
<span class="line"><span>1 device:</span></span>
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.391 GiB / 4.750 GiB available)</span></span>
<span class="line"><span>┌ Warning: LuxAMDGPU is loaded but the AMDGPU is not functional.</span></span>
<span class="line"><span>└ @ LuxAMDGPU ~/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/LuxAMDGPU/sGa0S/src/LuxAMDGPU.jl:19</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45),h=[l];function e(t,k,c,o,E,r){return n(),a("div",null,h)}const g=s(p,[["render",e]]);export{y as __pageData,g as default};
