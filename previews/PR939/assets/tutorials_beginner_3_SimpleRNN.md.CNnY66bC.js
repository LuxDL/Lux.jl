import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.XtOZMSQI.js";const r=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function e(h,s,t,c,k,o){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractLuxContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/previews/PR939/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Lux.AbstractLuxContainerLayer{(:lstm_cell, :classifier)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/previews/PR939/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/previews/PR939/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Main.var&quot;##225&quot;.SpiralClassifier</span></span></code></pre></div><p>We can use default Lux blocks – <code>Recurrence(LSTMCell(in_dims =&gt; hidden_dims)</code> – instead of defining the following. But let&#39;s still do it for the sake of it.</p><p>Now we need to define the behavior of the Classifier when it is invoked.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray{T, 3}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedTuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedTuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {T}</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # First we will have to run the sequence through the LSTM Cell</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # The first call to LSTM Cell will create the initial hidden state</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # See that the parameters and states are automatically populated into a field called</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # \`lstm_cell\` We use \`eachslice\` to get the elements in the sequence without copying,</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # and \`Iterators.peel\` to split out the first element for LSTM initialization.</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_init, x_rest </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Iterators</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">peel</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxOps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">eachslice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)))</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/previews/PR939/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; lstm_cell, classifier) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray{T, 3}</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {T}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        x_init, x_rest </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Iterators</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">peel</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxOps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">eachslice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        y, carry </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_init)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x_rest</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            y, carry </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x, carry))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vec</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>SpiralClassifierCompact (generic function with 1 method)</span></span></code></pre></div><h2 id="Defining-Accuracy,-Loss-and-Optimiser" tabindex="-1">Defining Accuracy, Loss and Optimiser <a class="header-anchor" href="#Defining-Accuracy,-Loss-and-Optimiser" aria-label="Permalink to &quot;Defining Accuracy, Loss and Optimiser {#Defining-Accuracy,-Loss-and-Optimiser}&quot;">​</a></h2><p>Now let&#39;s define the binarycrossentropy loss. Typically it is recommended to use <code>logitbinarycrossentropy</code> since it is more numerically stable, but for the sake of simplicity we will use <code>binarycrossentropy</code>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">const</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> lossfn </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> BinaryCrossEntropyLoss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> compute_loss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, (x, y))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ŷ, st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, ps, st)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lossfn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ŷ, y)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> loss, st_, (; y_pred</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ŷ)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">matches</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> sum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((y_pred </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.5f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.==</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y_true)</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> matches</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>accuracy (generic function with 1 method)</span></span></code></pre></div><h2 id="Training-the-Model" tabindex="-1">Training the Model <a class="header-anchor" href="#Training-the-Model" aria-label="Permalink to &quot;Training the Model {#Training-the-Model}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model_type)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    dev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> gpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Get the dataloaders</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_loader, val_loader </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">() </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model_type</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    rng </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Xoshiro</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ps, st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">setup</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng, model) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TrainState</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Adam</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.01f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">25</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Train the model</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> train_loader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            (_, loss, _, train_state) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">single_train_step!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                AutoZygote</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), lossfn, (x, y), train_state)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Epoch [%3d]: Loss %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch loss</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Validate the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> val_loader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            ŷ, st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, st_)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lossfn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ŷ, y)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ŷ, y)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Validation: Loss %4.5f Accuracy %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> loss acc</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> cpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.61340</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.58617</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.57155</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53297</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51268</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51050</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47083</span></span>
<span class="line"><span>Validation: Loss 0.48407 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46692 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46433</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46095</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43679</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41891</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41068</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39451</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38674</span></span>
<span class="line"><span>Validation: Loss 0.39098 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37036 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37795</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34914</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34473</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32375</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33412</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30401</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.29066</span></span>
<span class="line"><span>Validation: Loss 0.30834 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28611 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28981</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27138</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24894</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25852</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24375</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24456</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22885</span></span>
<span class="line"><span>Validation: Loss 0.23863 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21655 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20980</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20736</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19759</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19590</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17765</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18727</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.14508</span></span>
<span class="line"><span>Validation: Loss 0.18091 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16092 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15789</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15647</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15456</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13583</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13702</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12232</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13033</span></span>
<span class="line"><span>Validation: Loss 0.13450 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11820 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11785</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11467</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11137</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09695</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09410</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09981</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.07411</span></span>
<span class="line"><span>Validation: Loss 0.09657 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08464 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08749</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07058</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08045</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07428</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06399</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07101</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05612</span></span>
<span class="line"><span>Validation: Loss 0.06694 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05895 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05779</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05363</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05148</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04977</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05119</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05124</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03975</span></span>
<span class="line"><span>Validation: Loss 0.04925 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04357 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04344</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04058</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04237</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03999</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03780</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03454</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04272</span></span>
<span class="line"><span>Validation: Loss 0.03975 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03512 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03473</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03270</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03556</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02778</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03254</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03472</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02770</span></span>
<span class="line"><span>Validation: Loss 0.03372 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02971 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02734</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02903</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02791</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02858</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02919</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02653</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02668</span></span>
<span class="line"><span>Validation: Loss 0.02936 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02580 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02437</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02303</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02394</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02523</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02523</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02563</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02268</span></span>
<span class="line"><span>Validation: Loss 0.02600 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02278 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02109</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02172</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02319</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02203</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02108</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02171</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02023</span></span>
<span class="line"><span>Validation: Loss 0.02328 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02034 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02142</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01961</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01986</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01975</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01873</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01900</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01403</span></span>
<span class="line"><span>Validation: Loss 0.02104 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01833 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01829</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01820</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01676</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01836</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01779</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01641</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01716</span></span>
<span class="line"><span>Validation: Loss 0.01918 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01665 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01516</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01846</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01525</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01459</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01624</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01665</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01571</span></span>
<span class="line"><span>Validation: Loss 0.01759 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01524 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01595</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01591</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01617</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01320</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01420</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01290</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01498</span></span>
<span class="line"><span>Validation: Loss 0.01622 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01403 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01344</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01451</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01341</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01381</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01310</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01283</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01494</span></span>
<span class="line"><span>Validation: Loss 0.01503 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01298 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01308</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01230</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01280</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01342</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01153</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01229</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01289</span></span>
<span class="line"><span>Validation: Loss 0.01396 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01206 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01142</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01250</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01072</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01167</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01260</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01120</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01200</span></span>
<span class="line"><span>Validation: Loss 0.01301 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01123 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01154</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01075</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01175</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01087</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00934</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01107</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01118</span></span>
<span class="line"><span>Validation: Loss 0.01211 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01046 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01105</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00949</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01020</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01092</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01009</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00910</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00998</span></span>
<span class="line"><span>Validation: Loss 0.01120 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00969 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.01014</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00878</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00981</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00905</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00874</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00987</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00806</span></span>
<span class="line"><span>Validation: Loss 0.01019 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00884 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00891</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00928</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00847</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00825</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00853</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00808</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00571</span></span>
<span class="line"><span>Validation: Loss 0.00907 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00791 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62578</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.60226</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56476</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53133</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51075</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50488</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48660</span></span>
<span class="line"><span>Validation: Loss 0.47100 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46837 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46885</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45510</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43817</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42196</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41187</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39643</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39381</span></span>
<span class="line"><span>Validation: Loss 0.37378 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37061 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36666</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35038</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34060</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33479</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32448</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31618</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.29612</span></span>
<span class="line"><span>Validation: Loss 0.28845 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28512 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28814</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27163</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25609</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26036</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24986</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22906</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.20872</span></span>
<span class="line"><span>Validation: Loss 0.21803 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21492 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21709</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19812</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20066</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18830</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18050</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17827</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16019</span></span>
<span class="line"><span>Validation: Loss 0.16212 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.15952 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15572</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15123</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15348</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13785</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13134</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12747</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12255</span></span>
<span class="line"><span>Validation: Loss 0.11884 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11692 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11250</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11150</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10358</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10894</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10020</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08445</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09390</span></span>
<span class="line"><span>Validation: Loss 0.08482 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08354 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07888</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07495</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07859</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07761</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06450</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06721</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05048</span></span>
<span class="line"><span>Validation: Loss 0.05894 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05813 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06085</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04911</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05540</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05256</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04398</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04731</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04918</span></span>
<span class="line"><span>Validation: Loss 0.04392 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04328 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04705</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04174</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03934</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03853</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03710</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03633</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03240</span></span>
<span class="line"><span>Validation: Loss 0.03555 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03500 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03409</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03297</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03247</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03193</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03303</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03238</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03038</span></span>
<span class="line"><span>Validation: Loss 0.03023 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02974 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03245</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02884</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02778</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02653</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02761</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02664</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02365</span></span>
<span class="line"><span>Validation: Loss 0.02631 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02588 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02631</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02562</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02669</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02176</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02387</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02387</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02325</span></span>
<span class="line"><span>Validation: Loss 0.02330 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02291 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02231</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02211</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02177</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02200</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02128</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02227</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02005</span></span>
<span class="line"><span>Validation: Loss 0.02087 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02051 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02050</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02020</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01954</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01854</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01950</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02009</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01812</span></span>
<span class="line"><span>Validation: Loss 0.01885 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01852 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01935</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01857</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01784</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01940</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01558</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01567</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01961</span></span>
<span class="line"><span>Validation: Loss 0.01713 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01683 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01713</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01629</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01577</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01678</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01506</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01650</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01500</span></span>
<span class="line"><span>Validation: Loss 0.01566 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01538 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01639</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01373</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01399</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01538</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01463</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01460</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01597</span></span>
<span class="line"><span>Validation: Loss 0.01439 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01413 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01331</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01356</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01459</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01364</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01322</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01332</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01469</span></span>
<span class="line"><span>Validation: Loss 0.01327 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01303 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01377</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01249</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01222</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01173</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01305</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01229</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01263</span></span>
<span class="line"><span>Validation: Loss 0.01223 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01201 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01061</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01129</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01138</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01181</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01100</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01294</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01333</span></span>
<span class="line"><span>Validation: Loss 0.01120 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01101 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01058</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01105</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01090</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01064</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01039</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01020</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00891</span></span>
<span class="line"><span>Validation: Loss 0.01005 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00989 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00914</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00952</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01034</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00956</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00921</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00907</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00792</span></span>
<span class="line"><span>Validation: Loss 0.00889 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00876 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00858</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00860</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00854</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00879</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00770</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00796</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00855</span></span>
<span class="line"><span>Validation: Loss 0.00800 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00788 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00740</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00755</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00747</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00804</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00754</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00771</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00681</span></span>
<span class="line"><span>Validation: Loss 0.00736 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00724 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
<span class="line"><span> :ps_trained</span></span>
<span class="line"><span> :st_trained</span></span></code></pre></div><h2 id="appendix" tabindex="-1">Appendix <a class="header-anchor" href="#appendix" aria-label="Permalink to &quot;Appendix&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">InteractiveUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(MLDataDevices)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(CUDA) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&amp;&amp;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> MLDataDevices</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(CUDADevice)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        CUDA</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(AMDGPU) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&amp;&amp;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> MLDataDevices</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(AMDGPUDevice)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        AMDGPU</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Julia Version 1.10.5</span></span>
<span class="line"><span>Commit 6f3fdf7b362 (2024-08-27 14:19 UTC)</span></span>
<span class="line"><span>Build Info:</span></span>
<span class="line"><span>  Official https://julialang.org/ release</span></span>
<span class="line"><span>Platform Info:</span></span>
<span class="line"><span>  OS: Linux (x86_64-linux-gnu)</span></span>
<span class="line"><span>  CPU: 48 × AMD EPYC 7402 24-Core Processor</span></span>
<span class="line"><span>  WORD_SIZE: 64</span></span>
<span class="line"><span>  LIBM: libopenlibm</span></span>
<span class="line"><span>  LLVM: libLLVM-15.0.7 (ORCJIT, znver2)</span></span>
<span class="line"><span>Threads: 48 default, 0 interactive, 24 GC (on 2 virtual cores)</span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>  JULIA_CPU_THREADS = 2</span></span>
<span class="line"><span>  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6</span></span>
<span class="line"><span>  LD_LIBRARY_PATH = /usr/local/nvidia/lib:/usr/local/nvidia/lib64</span></span>
<span class="line"><span>  JULIA_PKG_SERVER = </span></span>
<span class="line"><span>  JULIA_NUM_THREADS = 48</span></span>
<span class="line"><span>  JULIA_CUDA_HARD_MEMORY_LIMIT = 100%</span></span>
<span class="line"><span>  JULIA_PKG_PRECOMPILE_AUTO = 0</span></span>
<span class="line"><span>  JULIA_DEBUG = Literate</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA runtime 12.5, artifact installation</span></span>
<span class="line"><span>CUDA driver 12.5</span></span>
<span class="line"><span>NVIDIA driver 555.42.6</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA libraries: </span></span>
<span class="line"><span>- CUBLAS: 12.5.3</span></span>
<span class="line"><span>- CURAND: 10.3.6</span></span>
<span class="line"><span>- CUFFT: 11.2.3</span></span>
<span class="line"><span>- CUSOLVER: 11.6.3</span></span>
<span class="line"><span>- CUSPARSE: 12.5.1</span></span>
<span class="line"><span>- CUPTI: 2024.2.1 (API 23.0.0)</span></span>
<span class="line"><span>- NVML: 12.0.0+555.42.6</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Julia packages: </span></span>
<span class="line"><span>- CUDA: 5.4.3</span></span>
<span class="line"><span>- CUDA_Driver_jll: 0.9.2+0</span></span>
<span class="line"><span>- CUDA_Runtime_jll: 0.14.1+0</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Toolchain:</span></span>
<span class="line"><span>- Julia: 1.10.5</span></span>
<span class="line"><span>- LLVM: 15.0.7</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>- JULIA_CUDA_HARD_MEMORY_LIMIT: 100%</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Preferences:</span></span>
<span class="line"><span>- CUDA_Driver_jll.compat: false</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2 devices:</span></span>
<span class="line"><span>  0: Quadro RTX 5000 (sm_75, 15.234 GiB / 16.000 GiB available)</span></span>
<span class="line"><span>  1: Quadro RTX 5000 (sm_75, 15.556 GiB / 16.000 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45)]))}const d=a(l,[["render",e]]);export{r as __pageData,d as default};
