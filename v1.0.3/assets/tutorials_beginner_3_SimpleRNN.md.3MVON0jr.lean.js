import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.Duzx1I2n.js";const r=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function h(e,s,t,c,k,o){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractLuxContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/v1.0.3/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Lux.AbstractLuxContainerLayer{(:lstm_cell, :classifier)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/v1.0.3/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/v1.0.3/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/v1.0.3/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62553</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.60510</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.55835</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54149</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.52378</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49478</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48398</span></span>
<span class="line"><span>Validation: Loss 0.46964 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.45746 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47575</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45342</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43241</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43282</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41279</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39988</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37074</span></span>
<span class="line"><span>Validation: Loss 0.37208 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.35773 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37759</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35141</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34756</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32144</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32601</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31558</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31383</span></span>
<span class="line"><span>Validation: Loss 0.28696 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.27097 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28812</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26925</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26329</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25165</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24186</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24788</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23699</span></span>
<span class="line"><span>Validation: Loss 0.21699 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.20103 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20486</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19917</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21229</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19976</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17600</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18594</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.15670</span></span>
<span class="line"><span>Validation: Loss 0.16125 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.14669 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16396</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15604</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14779</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14857</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12744</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12807</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.11374</span></span>
<span class="line"><span>Validation: Loss 0.11812 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.10615 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11222</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10495</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10957</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10119</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10683</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09341</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09857</span></span>
<span class="line"><span>Validation: Loss 0.08470 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.07579 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08797</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07320</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07783</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07383</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06510</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07110</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05521</span></span>
<span class="line"><span>Validation: Loss 0.05896 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05305 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05691</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05191</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05573</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05292</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04957</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04796</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04510</span></span>
<span class="line"><span>Validation: Loss 0.04381 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03964 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.05030</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03891</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04128</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03873</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03725</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03647</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03616</span></span>
<span class="line"><span>Validation: Loss 0.03545 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03206 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03576</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03512</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03400</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03295</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03057</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03171</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03048</span></span>
<span class="line"><span>Validation: Loss 0.03011 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02716 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02811</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03051</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02985</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02784</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02848</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02678</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02645</span></span>
<span class="line"><span>Validation: Loss 0.02622 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02360 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02600</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02664</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02552</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02399</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02336</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02505</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02253</span></span>
<span class="line"><span>Validation: Loss 0.02321 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02084 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02242</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02084</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02346</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02310</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02043</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02277</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02318</span></span>
<span class="line"><span>Validation: Loss 0.02080 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01863 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02136</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01921</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02144</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02043</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01801</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01950</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01957</span></span>
<span class="line"><span>Validation: Loss 0.01878 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01679 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01963</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01869</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01693</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01735</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01941</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01672</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01712</span></span>
<span class="line"><span>Validation: Loss 0.01707 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01522 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01644</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01655</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01663</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01688</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01635</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01556</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01802</span></span>
<span class="line"><span>Validation: Loss 0.01561 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01389 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01591</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01498</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01638</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01533</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01457</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01398</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01269</span></span>
<span class="line"><span>Validation: Loss 0.01433 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01273 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01489</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01322</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01360</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01345</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01382</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01379</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01584</span></span>
<span class="line"><span>Validation: Loss 0.01323 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01173 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01413</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01346</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01243</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01341</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01142</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01238</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01175</span></span>
<span class="line"><span>Validation: Loss 0.01222 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01083 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01175</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01221</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01252</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01250</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01099</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01107</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01175</span></span>
<span class="line"><span>Validation: Loss 0.01125 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00997 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01070</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01111</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01011</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01072</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01068</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01163</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01102</span></span>
<span class="line"><span>Validation: Loss 0.01020 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00906 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01049</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01014</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01008</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00944</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00962</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00895</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00913</span></span>
<span class="line"><span>Validation: Loss 0.00905 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00807 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00915</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00921</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00845</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00845</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00890</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00803</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00752</span></span>
<span class="line"><span>Validation: Loss 0.00808 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00723 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00786</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00768</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00805</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00816</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00758</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00742</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00741</span></span>
<span class="line"><span>Validation: Loss 0.00738 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00663 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.63003</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59696</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56691</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54274</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51708</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48755</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48718</span></span>
<span class="line"><span>Validation: Loss 0.46947 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46928 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46952</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46210</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43708</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41283</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40757</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39893</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40373</span></span>
<span class="line"><span>Validation: Loss 0.37232 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37178 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37683</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35058</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35233</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33516</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31468</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30104</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30092</span></span>
<span class="line"><span>Validation: Loss 0.28742 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28695 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28437</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26257</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25741</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26158</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24157</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23809</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25725</span></span>
<span class="line"><span>Validation: Loss 0.21740 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21727 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.22413</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20429</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19420</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19286</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17194</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17953</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16164</span></span>
<span class="line"><span>Validation: Loss 0.16148 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16178 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15849</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16297</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14785</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13243</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13312</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12825</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.11395</span></span>
<span class="line"><span>Validation: Loss 0.11824 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11879 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11945</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10755</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10389</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10288</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09822</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09268</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09171</span></span>
<span class="line"><span>Validation: Loss 0.08467 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08524 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08380</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07620</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06893</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07261</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07318</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06899</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05518</span></span>
<span class="line"><span>Validation: Loss 0.05907 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05948 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05651</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05527</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05700</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04890</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04720</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04867</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04540</span></span>
<span class="line"><span>Validation: Loss 0.04413 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04438 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04195</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04340</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04065</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04005</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03923</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03813</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03156</span></span>
<span class="line"><span>Validation: Loss 0.03586 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03605 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03494</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03379</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03358</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03390</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03312</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03158</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02879</span></span>
<span class="line"><span>Validation: Loss 0.03052 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03070 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03313</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02708</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02802</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02921</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02716</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02787</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02590</span></span>
<span class="line"><span>Validation: Loss 0.02661 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02677 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02568</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02455</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02725</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02433</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02632</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02308</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02287</span></span>
<span class="line"><span>Validation: Loss 0.02358 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02373 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02295</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02235</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02141</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02329</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02351</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02120</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01971</span></span>
<span class="line"><span>Validation: Loss 0.02114 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02127 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01979</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02159</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02015</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01984</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01975</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02058</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01560</span></span>
<span class="line"><span>Validation: Loss 0.01911 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01924 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01894</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01885</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01842</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01818</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01741</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01752</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01811</span></span>
<span class="line"><span>Validation: Loss 0.01740 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01751 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01640</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01702</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01696</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01563</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01621</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01695</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01844</span></span>
<span class="line"><span>Validation: Loss 0.01592 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01603 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01453</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01511</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01629</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01505</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01591</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01426</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01578</span></span>
<span class="line"><span>Validation: Loss 0.01463 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01473 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01349</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01488</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01332</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01482</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01563</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01231</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01245</span></span>
<span class="line"><span>Validation: Loss 0.01350 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01360 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01351</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01323</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01347</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01267</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01344</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01164</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01201</span></span>
<span class="line"><span>Validation: Loss 0.01250 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01260 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01225</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01262</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01212</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01126</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01137</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01230</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01205</span></span>
<span class="line"><span>Validation: Loss 0.01157 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01166 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01225</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01110</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01040</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01072</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01101</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01100</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01066</span></span>
<span class="line"><span>Validation: Loss 0.01059 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01068 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01040</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00974</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01038</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01026</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00959</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00992</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01033</span></span>
<span class="line"><span>Validation: Loss 0.00950 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00958 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00904</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.01043</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00946</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00901</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00827</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00763</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00912</span></span>
<span class="line"><span>Validation: Loss 0.00841 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00848 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00873</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00897</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00763</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00795</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00742</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00765</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00652</span></span>
<span class="line"><span>Validation: Loss 0.00760 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00765 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span>1 device:</span></span>
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.453 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45)]))}const d=a(l,[["render",h]]);export{r as __pageData,d as default};
