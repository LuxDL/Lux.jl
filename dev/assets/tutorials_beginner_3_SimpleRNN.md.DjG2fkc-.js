import{_ as a,c as n,o as e,al as i}from"./chunks/framework.BCN3FD2k.js";const k=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),p={name:"tutorials/beginner/3_SimpleRNN.md"};function t(c,s,l,r,h,f){return e(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><p>Note: If you wish to use AutoZygote() for automatic differentiation, add Zygote to your project dependencies and include <code>using Zygote</code>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, JLD2, MLUtils, Optimisers, Printf, Reactant, Random</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create the spirals</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [MLUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Datasets</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">make_spiral</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(sequence_length) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> _ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">dataset_size]</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Get the labels</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    labels </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vcat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">repeat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">([</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">repeat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">([</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    clockwise_spirals </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(d[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">][:, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">sequence_length], :, sequence_length, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        d </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    anticlockwise_spirals </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(d[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">][:, (sequence_length </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], :, sequence_length, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        d </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data[((dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Float32</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">cat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(clockwise_spirals</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, anticlockwise_spirals</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; dims</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Split the dataset</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (x_train, y_train), (x_val, y_val) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> splitobs</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x_data, labels); at</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create DataLoaders</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Use DataLoader to automatically minibatch and shuffle the data</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            collect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.((x_train, y_train)); batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, partial</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        ),</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Don&#39;t shuffle the validation data</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">collect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.((x_val, y_val)); batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, partial</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractLuxContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/dev/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L,C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> AbstractLuxContainerLayer{(:lstm_cell, :classifier)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/dev/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/dev/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Main.var&quot;##230&quot;.SpiralClassifier</span></span></code></pre></div><p>We can use default Lux blocks – <code>Recurrence(LSTMCell(in_dims =&gt; hidden_dims)</code> – instead of defining the following. But let&#39;s still do it for the sake of it.</p><p>Now we need to define the behavior of the Classifier when it is invoked.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray{T,3}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedTuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedTuple</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {T}</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/dev/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; lstm_cell, classifier) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray{T,3}</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {T}</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    dev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> reactant_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    cdev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> cpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Get the dataloaders</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_loader, val_loader </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> dev</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">())</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model_type</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ps, st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> dev</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">setup</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Random</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">default_rng</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), model))</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TrainState</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Adam</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.01f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    model_compiled </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> if</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">isa</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ReactantDevice</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @compile</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">first</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_loader)[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], ps, Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    else</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        model</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ad </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">isa</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ReactantDevice </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">?</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> AutoEnzyme</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">() </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> AutoZygote</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">25</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Train the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        total_loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.0f0</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        total_samples </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> train_loader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            (_, loss, _, train_state) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">single_train_step!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                ad, lossfn, (x, y), train_state</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            )</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            total_loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            total_samples </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Epoch [%3d]: Loss %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch (total_loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> total_samples)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Validate the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        total_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.0f0</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        total_loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.0f0</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        total_samples </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> val_loader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            ŷ, st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model_compiled</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, st_)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            ŷ, y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> cdev</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ŷ), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">cdev</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            total_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ŷ, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            total_loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lossfn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ŷ, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            total_samples </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Validation:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">Loss %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">Accuracy %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (total_loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> total_samples) (</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            total_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> total_samples</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        )</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> cpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()((train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>┌ Warning: \`replicate\` doesn&#39;t work for \`TaskLocalRNG\`. Returning the same \`TaskLocalRNG\`.</span></span>
<span class="line"><span>└ @ LuxCore /var/lib/buildkite-agent/builds/gpuci-10/julialang/lux-dot-jl/lib/LuxCore/src/LuxCore.jl:18</span></span>
<span class="line"><span>2025-05-08 13:32:16.382854: I external/xla/xla/service/service.cc:152] XLA service 0x2369b6f0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:</span></span>
<span class="line"><span>2025-05-08 13:32:16.383183: I external/xla/xla/service/service.cc:160]   StreamExecutor device (0): NVIDIA A100-PCIE-40GB MIG 1g.5gb, Compute Capability 8.0</span></span>
<span class="line"><span>WARNING: All log messages before absl::InitializeLog() is called are written to STDERR</span></span>
<span class="line"><span>I0000 00:00:1746711136.383890 3123193 se_gpu_pjrt_client.cc:1026] Using BFC allocator.</span></span>
<span class="line"><span>I0000 00:00:1746711136.383965 3123193 gpu_helpers.cc:136] XLA backend allocating 3825205248 bytes on device 0 for BFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1746711136.383997 3123193 gpu_helpers.cc:177] XLA backend will use up to 1275068416 bytes on device 0 for CollectiveBFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1746711136.399874 3123193 cuda_dnn.cc:529] Loaded cuDNN version 90400</span></span>
<span class="line"><span>E0000 00:00:1746711197.044877 3123193 buffer_comparator.cc:145] Difference at 16: 0, expected 0.966326</span></span>
<span class="line"><span>E0000 00:00:1746711197.044930 3123193 buffer_comparator.cc:145] Difference at 17: 0, expected 0.955446</span></span>
<span class="line"><span>E0000 00:00:1746711197.044941 3123193 buffer_comparator.cc:145] Difference at 18: 0, expected 0.522552</span></span>
<span class="line"><span>E0000 00:00:1746711197.044944 3123193 buffer_comparator.cc:145] Difference at 19: 0, expected 0.554959</span></span>
<span class="line"><span>E0000 00:00:1746711197.044947 3123193 buffer_comparator.cc:145] Difference at 20: 0, expected 0.833471</span></span>
<span class="line"><span>E0000 00:00:1746711197.044950 3123193 buffer_comparator.cc:145] Difference at 21: 0, expected 0.404081</span></span>
<span class="line"><span>E0000 00:00:1746711197.044953 3123193 buffer_comparator.cc:145] Difference at 22: 0, expected 0.289287</span></span>
<span class="line"><span>E0000 00:00:1746711197.044956 3123193 buffer_comparator.cc:145] Difference at 23: 0, expected 0.732437</span></span>
<span class="line"><span>E0000 00:00:1746711197.044958 3123193 buffer_comparator.cc:145] Difference at 24: 0, expected 1.02391</span></span>
<span class="line"><span>E0000 00:00:1746711197.044961 3123193 buffer_comparator.cc:145] Difference at 25: 0, expected 0.647103</span></span>
<span class="line"><span>2025-05-08 13:33:17.044972: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746711197.047637 3123193 buffer_comparator.cc:145] Difference at 16: 0, expected 0.966326</span></span>
<span class="line"><span>E0000 00:00:1746711197.047649 3123193 buffer_comparator.cc:145] Difference at 17: 0, expected 0.955446</span></span>
<span class="line"><span>E0000 00:00:1746711197.047653 3123193 buffer_comparator.cc:145] Difference at 18: 0, expected 0.522552</span></span>
<span class="line"><span>E0000 00:00:1746711197.047657 3123193 buffer_comparator.cc:145] Difference at 19: 0, expected 0.554959</span></span>
<span class="line"><span>E0000 00:00:1746711197.047660 3123193 buffer_comparator.cc:145] Difference at 20: 0, expected 0.833471</span></span>
<span class="line"><span>E0000 00:00:1746711197.047663 3123193 buffer_comparator.cc:145] Difference at 21: 0, expected 0.404081</span></span>
<span class="line"><span>E0000 00:00:1746711197.047666 3123193 buffer_comparator.cc:145] Difference at 22: 0, expected 0.289287</span></span>
<span class="line"><span>E0000 00:00:1746711197.047669 3123193 buffer_comparator.cc:145] Difference at 23: 0, expected 0.732437</span></span>
<span class="line"><span>E0000 00:00:1746711197.047671 3123193 buffer_comparator.cc:145] Difference at 24: 0, expected 1.02391</span></span>
<span class="line"><span>E0000 00:00:1746711197.047674 3123193 buffer_comparator.cc:145] Difference at 25: 0, expected 0.647103</span></span>
<span class="line"><span>2025-05-08 13:33:17.047680: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746711197.050069 3123193 buffer_comparator.cc:145] Difference at 16: 0, expected 0.966326</span></span>
<span class="line"><span>E0000 00:00:1746711197.050083 3123193 buffer_comparator.cc:145] Difference at 17: 0, expected 0.955446</span></span>
<span class="line"><span>E0000 00:00:1746711197.050088 3123193 buffer_comparator.cc:145] Difference at 18: 0, expected 0.522552</span></span>
<span class="line"><span>E0000 00:00:1746711197.050092 3123193 buffer_comparator.cc:145] Difference at 19: 0, expected 0.554959</span></span>
<span class="line"><span>E0000 00:00:1746711197.050096 3123193 buffer_comparator.cc:145] Difference at 20: 0, expected 0.833471</span></span>
<span class="line"><span>E0000 00:00:1746711197.050099 3123193 buffer_comparator.cc:145] Difference at 21: 0, expected 0.404081</span></span>
<span class="line"><span>E0000 00:00:1746711197.050102 3123193 buffer_comparator.cc:145] Difference at 22: 0, expected 0.289287</span></span>
<span class="line"><span>E0000 00:00:1746711197.050105 3123193 buffer_comparator.cc:145] Difference at 23: 0, expected 0.732437</span></span>
<span class="line"><span>E0000 00:00:1746711197.050109 3123193 buffer_comparator.cc:145] Difference at 24: 0, expected 1.02391</span></span>
<span class="line"><span>E0000 00:00:1746711197.050112 3123193 buffer_comparator.cc:145] Difference at 25: 0, expected 0.647103</span></span>
<span class="line"><span>2025-05-08 13:33:17.050117: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746711197.052505 3123193 buffer_comparator.cc:145] Difference at 16: 0, expected 0.966326</span></span>
<span class="line"><span>E0000 00:00:1746711197.052518 3123193 buffer_comparator.cc:145] Difference at 17: 0, expected 0.955446</span></span>
<span class="line"><span>E0000 00:00:1746711197.052522 3123193 buffer_comparator.cc:145] Difference at 18: 0, expected 0.522552</span></span>
<span class="line"><span>E0000 00:00:1746711197.052525 3123193 buffer_comparator.cc:145] Difference at 19: 0, expected 0.554959</span></span>
<span class="line"><span>E0000 00:00:1746711197.052528 3123193 buffer_comparator.cc:145] Difference at 20: 0, expected 0.833471</span></span>
<span class="line"><span>E0000 00:00:1746711197.052531 3123193 buffer_comparator.cc:145] Difference at 21: 0, expected 0.404081</span></span>
<span class="line"><span>E0000 00:00:1746711197.052534 3123193 buffer_comparator.cc:145] Difference at 22: 0, expected 0.289287</span></span>
<span class="line"><span>E0000 00:00:1746711197.052537 3123193 buffer_comparator.cc:145] Difference at 23: 0, expected 0.732437</span></span>
<span class="line"><span>E0000 00:00:1746711197.052540 3123193 buffer_comparator.cc:145] Difference at 24: 0, expected 1.02391</span></span>
<span class="line"><span>E0000 00:00:1746711197.052543 3123193 buffer_comparator.cc:145] Difference at 25: 0, expected 0.647103</span></span>
<span class="line"><span>2025-05-08 13:33:17.052548: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746711197.054926 3123193 buffer_comparator.cc:145] Difference at 32: 0, expected 0.904315</span></span>
<span class="line"><span>E0000 00:00:1746711197.054938 3123193 buffer_comparator.cc:145] Difference at 33: 0, expected 1.02658</span></span>
<span class="line"><span>E0000 00:00:1746711197.054942 3123193 buffer_comparator.cc:145] Difference at 34: 0, expected 0.512492</span></span>
<span class="line"><span>E0000 00:00:1746711197.054946 3123193 buffer_comparator.cc:145] Difference at 35: 0, expected 0.434209</span></span>
<span class="line"><span>E0000 00:00:1746711197.054949 3123193 buffer_comparator.cc:145] Difference at 36: 0, expected 0.218704</span></span>
<span class="line"><span>E0000 00:00:1746711197.054951 3123193 buffer_comparator.cc:145] Difference at 37: 0, expected 0.551313</span></span>
<span class="line"><span>E0000 00:00:1746711197.054954 3123193 buffer_comparator.cc:145] Difference at 38: 0, expected 1.10187</span></span>
<span class="line"><span>E0000 00:00:1746711197.054957 3123193 buffer_comparator.cc:145] Difference at 39: 0, expected 0.347384</span></span>
<span class="line"><span>E0000 00:00:1746711197.054960 3123193 buffer_comparator.cc:145] Difference at 40: 0, expected 0.789874</span></span>
<span class="line"><span>E0000 00:00:1746711197.054963 3123193 buffer_comparator.cc:145] Difference at 41: 0, expected 0.204116</span></span>
<span class="line"><span>2025-05-08 13:33:17.054968: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746711197.057565 3123193 buffer_comparator.cc:145] Difference at 32: 0, expected 0.904315</span></span>
<span class="line"><span>E0000 00:00:1746711197.057577 3123193 buffer_comparator.cc:145] Difference at 33: 0, expected 1.02658</span></span>
<span class="line"><span>E0000 00:00:1746711197.057581 3123193 buffer_comparator.cc:145] Difference at 34: 0, expected 0.512492</span></span>
<span class="line"><span>E0000 00:00:1746711197.057584 3123193 buffer_comparator.cc:145] Difference at 35: 0, expected 0.434209</span></span>
<span class="line"><span>E0000 00:00:1746711197.057587 3123193 buffer_comparator.cc:145] Difference at 36: 0, expected 0.218704</span></span>
<span class="line"><span>E0000 00:00:1746711197.057590 3123193 buffer_comparator.cc:145] Difference at 37: 0, expected 0.551313</span></span>
<span class="line"><span>E0000 00:00:1746711197.057593 3123193 buffer_comparator.cc:145] Difference at 38: 0, expected 1.10187</span></span>
<span class="line"><span>E0000 00:00:1746711197.057596 3123193 buffer_comparator.cc:145] Difference at 39: 0, expected 0.347384</span></span>
<span class="line"><span>E0000 00:00:1746711197.057599 3123193 buffer_comparator.cc:145] Difference at 40: 0, expected 0.789874</span></span>
<span class="line"><span>E0000 00:00:1746711197.057602 3123193 buffer_comparator.cc:145] Difference at 41: 0, expected 0.204116</span></span>
<span class="line"><span>2025-05-08 13:33:17.057606: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746711197.060000 3123193 buffer_comparator.cc:145] Difference at 32: 0, expected 0.904315</span></span>
<span class="line"><span>E0000 00:00:1746711197.060013 3123193 buffer_comparator.cc:145] Difference at 33: 0, expected 1.02658</span></span>
<span class="line"><span>E0000 00:00:1746711197.060017 3123193 buffer_comparator.cc:145] Difference at 34: 0, expected 0.512492</span></span>
<span class="line"><span>E0000 00:00:1746711197.060020 3123193 buffer_comparator.cc:145] Difference at 35: 0, expected 0.434209</span></span>
<span class="line"><span>E0000 00:00:1746711197.060023 3123193 buffer_comparator.cc:145] Difference at 36: 0, expected 0.218704</span></span>
<span class="line"><span>E0000 00:00:1746711197.060026 3123193 buffer_comparator.cc:145] Difference at 37: 0, expected 0.551313</span></span>
<span class="line"><span>E0000 00:00:1746711197.060029 3123193 buffer_comparator.cc:145] Difference at 38: 0, expected 1.10187</span></span>
<span class="line"><span>E0000 00:00:1746711197.060032 3123193 buffer_comparator.cc:145] Difference at 39: 0, expected 0.347384</span></span>
<span class="line"><span>E0000 00:00:1746711197.060035 3123193 buffer_comparator.cc:145] Difference at 40: 0, expected 0.789874</span></span>
<span class="line"><span>E0000 00:00:1746711197.060037 3123193 buffer_comparator.cc:145] Difference at 41: 0, expected 0.204116</span></span>
<span class="line"><span>2025-05-08 13:33:17.060042: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746711197.062624 3123193 buffer_comparator.cc:145] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1746711197.062637 3123193 buffer_comparator.cc:145] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1746711197.062641 3123193 buffer_comparator.cc:145] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1746711197.062644 3123193 buffer_comparator.cc:145] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1746711197.062647 3123193 buffer_comparator.cc:145] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1746711197.062650 3123193 buffer_comparator.cc:145] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1746711197.062653 3123193 buffer_comparator.cc:145] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1746711197.062656 3123193 buffer_comparator.cc:145] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1746711197.062658 3123193 buffer_comparator.cc:145] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1746711197.062661 3123193 buffer_comparator.cc:145] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-05-08 13:33:17.062666: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746711197.065058 3123193 buffer_comparator.cc:145] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1746711197.065071 3123193 buffer_comparator.cc:145] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1746711197.065075 3123193 buffer_comparator.cc:145] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1746711197.065078 3123193 buffer_comparator.cc:145] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1746711197.065081 3123193 buffer_comparator.cc:145] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1746711197.065084 3123193 buffer_comparator.cc:145] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1746711197.065087 3123193 buffer_comparator.cc:145] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1746711197.065090 3123193 buffer_comparator.cc:145] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1746711197.065093 3123193 buffer_comparator.cc:145] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1746711197.065096 3123193 buffer_comparator.cc:145] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-05-08 13:33:17.065101: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746711197.067491 3123193 buffer_comparator.cc:145] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1746711197.067504 3123193 buffer_comparator.cc:145] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1746711197.067508 3123193 buffer_comparator.cc:145] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1746711197.067512 3123193 buffer_comparator.cc:145] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1746711197.067515 3123193 buffer_comparator.cc:145] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1746711197.067518 3123193 buffer_comparator.cc:145] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1746711197.067521 3123193 buffer_comparator.cc:145] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1746711197.067524 3123193 buffer_comparator.cc:145] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1746711197.067527 3123193 buffer_comparator.cc:145] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1746711197.067530 3123193 buffer_comparator.cc:145] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-05-08 13:33:17.067534: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746711197.069926 3123193 buffer_comparator.cc:145] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1746711197.069939 3123193 buffer_comparator.cc:145] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1746711197.069943 3123193 buffer_comparator.cc:145] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1746711197.069946 3123193 buffer_comparator.cc:145] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1746711197.069949 3123193 buffer_comparator.cc:145] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1746711197.069952 3123193 buffer_comparator.cc:145] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1746711197.069955 3123193 buffer_comparator.cc:145] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1746711197.069958 3123193 buffer_comparator.cc:145] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1746711197.069961 3123193 buffer_comparator.cc:145] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1746711197.069964 3123193 buffer_comparator.cc:145] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-05-08 13:33:17.069968: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746711197.072355 3123193 buffer_comparator.cc:145] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1746711197.072367 3123193 buffer_comparator.cc:145] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1746711197.072371 3123193 buffer_comparator.cc:145] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1746711197.072374 3123193 buffer_comparator.cc:145] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1746711197.072377 3123193 buffer_comparator.cc:145] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1746711197.072380 3123193 buffer_comparator.cc:145] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1746711197.072383 3123193 buffer_comparator.cc:145] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1746711197.072386 3123193 buffer_comparator.cc:145] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1746711197.072389 3123193 buffer_comparator.cc:145] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1746711197.072392 3123193 buffer_comparator.cc:145] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-05-08 13:33:17.072397: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746711197.074787 3123193 buffer_comparator.cc:145] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1746711197.074799 3123193 buffer_comparator.cc:145] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1746711197.074803 3123193 buffer_comparator.cc:145] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1746711197.074806 3123193 buffer_comparator.cc:145] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1746711197.074809 3123193 buffer_comparator.cc:145] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1746711197.074812 3123193 buffer_comparator.cc:145] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1746711197.074815 3123193 buffer_comparator.cc:145] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1746711197.074819 3123193 buffer_comparator.cc:145] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1746711197.074822 3123193 buffer_comparator.cc:145] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1746711197.074825 3123193 buffer_comparator.cc:145] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-05-08 13:33:17.074830: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746711197.077229 3123193 buffer_comparator.cc:145] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1746711197.077242 3123193 buffer_comparator.cc:145] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1746711197.077246 3123193 buffer_comparator.cc:145] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1746711197.077249 3123193 buffer_comparator.cc:145] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1746711197.077252 3123193 buffer_comparator.cc:145] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1746711197.077255 3123193 buffer_comparator.cc:145] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1746711197.077257 3123193 buffer_comparator.cc:145] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1746711197.077260 3123193 buffer_comparator.cc:145] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1746711197.077263 3123193 buffer_comparator.cc:145] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1746711197.077266 3123193 buffer_comparator.cc:145] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-05-08 13:33:17.077271: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746711197.079668 3123193 buffer_comparator.cc:145] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1746711197.079681 3123193 buffer_comparator.cc:145] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1746711197.079685 3123193 buffer_comparator.cc:145] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1746711197.079688 3123193 buffer_comparator.cc:145] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1746711197.079690 3123193 buffer_comparator.cc:145] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1746711197.079693 3123193 buffer_comparator.cc:145] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1746711197.079696 3123193 buffer_comparator.cc:145] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1746711197.079699 3123193 buffer_comparator.cc:145] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1746711197.079702 3123193 buffer_comparator.cc:145] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1746711197.079705 3123193 buffer_comparator.cc:145] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-05-08 13:33:17.079710: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746711197.082197 3123193 buffer_comparator.cc:145] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1746711197.082209 3123193 buffer_comparator.cc:145] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1746711197.082213 3123193 buffer_comparator.cc:145] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1746711197.082216 3123193 buffer_comparator.cc:145] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1746711197.082219 3123193 buffer_comparator.cc:145] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1746711197.082222 3123193 buffer_comparator.cc:145] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1746711197.082225 3123193 buffer_comparator.cc:145] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1746711197.082228 3123193 buffer_comparator.cc:145] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1746711197.082231 3123193 buffer_comparator.cc:145] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1746711197.082234 3123193 buffer_comparator.cc:145] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-05-08 13:33:17.082239: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746711197.084635 3123193 buffer_comparator.cc:145] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1746711197.084648 3123193 buffer_comparator.cc:145] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1746711197.084652 3123193 buffer_comparator.cc:145] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1746711197.084655 3123193 buffer_comparator.cc:145] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1746711197.084657 3123193 buffer_comparator.cc:145] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1746711197.084660 3123193 buffer_comparator.cc:145] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1746711197.084663 3123193 buffer_comparator.cc:145] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1746711197.084666 3123193 buffer_comparator.cc:145] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1746711197.084669 3123193 buffer_comparator.cc:145] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1746711197.084672 3123193 buffer_comparator.cc:145] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-05-08 13:33:17.084677: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746711197.087239 3123193 buffer_comparator.cc:145] Difference at 64: 0, expected 0.629991</span></span>
<span class="line"><span>E0000 00:00:1746711197.087251 3123193 buffer_comparator.cc:145] Difference at 65: 0, expected 0.54577</span></span>
<span class="line"><span>E0000 00:00:1746711197.087255 3123193 buffer_comparator.cc:145] Difference at 66: 0, expected 0.316298</span></span>
<span class="line"><span>E0000 00:00:1746711197.087258 3123193 buffer_comparator.cc:145] Difference at 67: 0, expected 0.438545</span></span>
<span class="line"><span>E0000 00:00:1746711197.087261 3123193 buffer_comparator.cc:145] Difference at 68: 0, expected 0.523314</span></span>
<span class="line"><span>E0000 00:00:1746711197.087264 3123193 buffer_comparator.cc:145] Difference at 69: 0, expected 0.83106</span></span>
<span class="line"><span>E0000 00:00:1746711197.087266 3123193 buffer_comparator.cc:145] Difference at 70: 0, expected 0.617399</span></span>
<span class="line"><span>E0000 00:00:1746711197.087269 3123193 buffer_comparator.cc:145] Difference at 71: 0, expected 0.692252</span></span>
<span class="line"><span>E0000 00:00:1746711197.087272 3123193 buffer_comparator.cc:145] Difference at 72: 0, expected 0.185378</span></span>
<span class="line"><span>E0000 00:00:1746711197.087275 3123193 buffer_comparator.cc:145] Difference at 73: 0, expected 0.689502</span></span>
<span class="line"><span>2025-05-08 13:33:17.087280: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746711197.089672 3123193 buffer_comparator.cc:145] Difference at 128: 0, expected 1.00573</span></span>
<span class="line"><span>E0000 00:00:1746711197.089684 3123193 buffer_comparator.cc:145] Difference at 129: 0, expected 0.406227</span></span>
<span class="line"><span>E0000 00:00:1746711197.089688 3123193 buffer_comparator.cc:145] Difference at 130: 0, expected 0.311948</span></span>
<span class="line"><span>E0000 00:00:1746711197.089691 3123193 buffer_comparator.cc:145] Difference at 131: 0, expected 0.53677</span></span>
<span class="line"><span>E0000 00:00:1746711197.089694 3123193 buffer_comparator.cc:145] Difference at 132: 0, expected 0.172814</span></span>
<span class="line"><span>E0000 00:00:1746711197.089697 3123193 buffer_comparator.cc:145] Difference at 133: 0, expected 0.314312</span></span>
<span class="line"><span>E0000 00:00:1746711197.089700 3123193 buffer_comparator.cc:145] Difference at 134: 0, expected 1.17027</span></span>
<span class="line"><span>E0000 00:00:1746711197.089703 3123193 buffer_comparator.cc:145] Difference at 135: 0, expected 1.05396</span></span>
<span class="line"><span>E0000 00:00:1746711197.089706 3123193 buffer_comparator.cc:145] Difference at 136: 0, expected 0.788122</span></span>
<span class="line"><span>E0000 00:00:1746711197.089709 3123193 buffer_comparator.cc:145] Difference at 137: 0, expected 0.232274</span></span>
<span class="line"><span>2025-05-08 13:33:17.089714: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746711197.092121 3123193 buffer_comparator.cc:145] Difference at 128: 0, expected 1.00573</span></span>
<span class="line"><span>E0000 00:00:1746711197.092133 3123193 buffer_comparator.cc:145] Difference at 129: 0, expected 0.406227</span></span>
<span class="line"><span>E0000 00:00:1746711197.092137 3123193 buffer_comparator.cc:145] Difference at 130: 0, expected 0.311948</span></span>
<span class="line"><span>E0000 00:00:1746711197.092141 3123193 buffer_comparator.cc:145] Difference at 131: 0, expected 0.53677</span></span>
<span class="line"><span>E0000 00:00:1746711197.092144 3123193 buffer_comparator.cc:145] Difference at 132: 0, expected 0.172814</span></span>
<span class="line"><span>E0000 00:00:1746711197.092147 3123193 buffer_comparator.cc:145] Difference at 133: 0, expected 0.314312</span></span>
<span class="line"><span>E0000 00:00:1746711197.092150 3123193 buffer_comparator.cc:145] Difference at 134: 0, expected 1.17027</span></span>
<span class="line"><span>E0000 00:00:1746711197.092153 3123193 buffer_comparator.cc:145] Difference at 135: 0, expected 1.05396</span></span>
<span class="line"><span>E0000 00:00:1746711197.092156 3123193 buffer_comparator.cc:145] Difference at 136: 0, expected 0.788122</span></span>
<span class="line"><span>E0000 00:00:1746711197.092159 3123193 buffer_comparator.cc:145] Difference at 137: 0, expected 0.232274</span></span>
<span class="line"><span>2025-05-08 13:33:17.092164: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746711197.094562 3123193 buffer_comparator.cc:145] Difference at 128: 0, expected 1.00573</span></span>
<span class="line"><span>E0000 00:00:1746711197.094574 3123193 buffer_comparator.cc:145] Difference at 129: 0, expected 0.406227</span></span>
<span class="line"><span>E0000 00:00:1746711197.094578 3123193 buffer_comparator.cc:145] Difference at 130: 0, expected 0.311948</span></span>
<span class="line"><span>E0000 00:00:1746711197.094581 3123193 buffer_comparator.cc:145] Difference at 131: 0, expected 0.53677</span></span>
<span class="line"><span>E0000 00:00:1746711197.094584 3123193 buffer_comparator.cc:145] Difference at 132: 0, expected 0.172814</span></span>
<span class="line"><span>E0000 00:00:1746711197.094587 3123193 buffer_comparator.cc:145] Difference at 133: 0, expected 0.314312</span></span>
<span class="line"><span>E0000 00:00:1746711197.094590 3123193 buffer_comparator.cc:145] Difference at 134: 0, expected 1.17027</span></span>
<span class="line"><span>E0000 00:00:1746711197.094593 3123193 buffer_comparator.cc:145] Difference at 135: 0, expected 1.05396</span></span>
<span class="line"><span>E0000 00:00:1746711197.094596 3123193 buffer_comparator.cc:145] Difference at 136: 0, expected 0.788122</span></span>
<span class="line"><span>E0000 00:00:1746711197.094599 3123193 buffer_comparator.cc:145] Difference at 137: 0, expected 0.232274</span></span>
<span class="line"><span>2025-05-08 13:33:17.094604: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746711197.097001 3123193 buffer_comparator.cc:145] Difference at 128: 0, expected 1.00573</span></span>
<span class="line"><span>E0000 00:00:1746711197.097014 3123193 buffer_comparator.cc:145] Difference at 129: 0, expected 0.406227</span></span>
<span class="line"><span>E0000 00:00:1746711197.097018 3123193 buffer_comparator.cc:145] Difference at 130: 0, expected 0.311948</span></span>
<span class="line"><span>E0000 00:00:1746711197.097021 3123193 buffer_comparator.cc:145] Difference at 131: 0, expected 0.53677</span></span>
<span class="line"><span>E0000 00:00:1746711197.097024 3123193 buffer_comparator.cc:145] Difference at 132: 0, expected 0.172814</span></span>
<span class="line"><span>E0000 00:00:1746711197.097027 3123193 buffer_comparator.cc:145] Difference at 133: 0, expected 0.314312</span></span>
<span class="line"><span>E0000 00:00:1746711197.097030 3123193 buffer_comparator.cc:145] Difference at 134: 0, expected 1.17027</span></span>
<span class="line"><span>E0000 00:00:1746711197.097033 3123193 buffer_comparator.cc:145] Difference at 135: 0, expected 1.05396</span></span>
<span class="line"><span>E0000 00:00:1746711197.097036 3123193 buffer_comparator.cc:145] Difference at 136: 0, expected 0.788122</span></span>
<span class="line"><span>E0000 00:00:1746711197.097039 3123193 buffer_comparator.cc:145] Difference at 137: 0, expected 0.232274</span></span>
<span class="line"><span>2025-05-08 13:33:17.097043: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746711197.099446 3123193 buffer_comparator.cc:145] Difference at 128: 0, expected 1.00573</span></span>
<span class="line"><span>E0000 00:00:1746711197.099458 3123193 buffer_comparator.cc:145] Difference at 129: 0, expected 0.406227</span></span>
<span class="line"><span>E0000 00:00:1746711197.099462 3123193 buffer_comparator.cc:145] Difference at 130: 0, expected 0.311948</span></span>
<span class="line"><span>E0000 00:00:1746711197.099465 3123193 buffer_comparator.cc:145] Difference at 131: 0, expected 0.53677</span></span>
<span class="line"><span>E0000 00:00:1746711197.099468 3123193 buffer_comparator.cc:145] Difference at 132: 0, expected 0.172814</span></span>
<span class="line"><span>E0000 00:00:1746711197.099471 3123193 buffer_comparator.cc:145] Difference at 133: 0, expected 0.314312</span></span>
<span class="line"><span>E0000 00:00:1746711197.099474 3123193 buffer_comparator.cc:145] Difference at 134: 0, expected 1.17027</span></span>
<span class="line"><span>E0000 00:00:1746711197.099478 3123193 buffer_comparator.cc:145] Difference at 135: 0, expected 1.05396</span></span>
<span class="line"><span>E0000 00:00:1746711197.099481 3123193 buffer_comparator.cc:145] Difference at 136: 0, expected 0.788122</span></span>
<span class="line"><span>E0000 00:00:1746711197.099484 3123193 buffer_comparator.cc:145] Difference at 137: 0, expected 0.232274</span></span>
<span class="line"><span>2025-05-08 13:33:17.099489: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746711197.102185 3123193 buffer_comparator.cc:145] Difference at 32: -nan, expected 1.62244</span></span>
<span class="line"><span>E0000 00:00:1746711197.102197 3123193 buffer_comparator.cc:145] Difference at 33: -nan, expected 1.87084</span></span>
<span class="line"><span>E0000 00:00:1746711197.102201 3123193 buffer_comparator.cc:145] Difference at 34: -nan, expected 1.07351</span></span>
<span class="line"><span>E0000 00:00:1746711197.102203 3123193 buffer_comparator.cc:145] Difference at 35: -nan, expected 2.92445</span></span>
<span class="line"><span>E0000 00:00:1746711197.102206 3123193 buffer_comparator.cc:145] Difference at 36: -nan, expected 1.98056</span></span>
<span class="line"><span>E0000 00:00:1746711197.102209 3123193 buffer_comparator.cc:145] Difference at 37: -nan, expected 2.07715</span></span>
<span class="line"><span>E0000 00:00:1746711197.102212 3123193 buffer_comparator.cc:145] Difference at 38: -nan, expected 1.56458</span></span>
<span class="line"><span>E0000 00:00:1746711197.102214 3123193 buffer_comparator.cc:145] Difference at 39: -nan, expected 2.27034</span></span>
<span class="line"><span>E0000 00:00:1746711197.102217 3123193 buffer_comparator.cc:145] Difference at 40: -nan, expected 2.31795</span></span>
<span class="line"><span>E0000 00:00:1746711197.102220 3123193 buffer_comparator.cc:145] Difference at 41: -nan, expected 2.55731</span></span>
<span class="line"><span>2025-05-08 13:33:17.102224: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746711234.777501 3123193 buffer_comparator.cc:145] Difference at 16: 0.196842, expected 9.68745</span></span>
<span class="line"><span>E0000 00:00:1746711234.777550 3123193 buffer_comparator.cc:145] Difference at 17: 0.688536, expected 10.1876</span></span>
<span class="line"><span>E0000 00:00:1746711234.777557 3123193 buffer_comparator.cc:145] Difference at 18: 0.927057, expected 8.84104</span></span>
<span class="line"><span>E0000 00:00:1746711234.777560 3123193 buffer_comparator.cc:145] Difference at 19: 0.579189, expected 10.0381</span></span>
<span class="line"><span>E0000 00:00:1746711234.777563 3123193 buffer_comparator.cc:145] Difference at 20: 0.374055, expected 7.30446</span></span>
<span class="line"><span>E0000 00:00:1746711234.777566 3123193 buffer_comparator.cc:145] Difference at 21: 0.216797, expected 8.26483</span></span>
<span class="line"><span>E0000 00:00:1746711234.777569 3123193 buffer_comparator.cc:145] Difference at 22: 0.731212, expected 10.8549</span></span>
<span class="line"><span>E0000 00:00:1746711234.777572 3123193 buffer_comparator.cc:145] Difference at 23: 0.700668, expected 7.87482</span></span>
<span class="line"><span>E0000 00:00:1746711234.777575 3123193 buffer_comparator.cc:145] Difference at 24: 0.5317, expected 9.78239</span></span>
<span class="line"><span>E0000 00:00:1746711234.777578 3123193 buffer_comparator.cc:145] Difference at 25: 0.24009, expected 11.3838</span></span>
<span class="line"><span>2025-05-08 13:33:54.777590: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746711234.779652 3123193 buffer_comparator.cc:145] Difference at 16: 0.196842, expected 9.68745</span></span>
<span class="line"><span>E0000 00:00:1746711234.779665 3123193 buffer_comparator.cc:145] Difference at 17: 0.688536, expected 10.1876</span></span>
<span class="line"><span>E0000 00:00:1746711234.779669 3123193 buffer_comparator.cc:145] Difference at 18: 0.927057, expected 8.84104</span></span>
<span class="line"><span>E0000 00:00:1746711234.779672 3123193 buffer_comparator.cc:145] Difference at 19: 0.579189, expected 10.0381</span></span>
<span class="line"><span>E0000 00:00:1746711234.779675 3123193 buffer_comparator.cc:145] Difference at 20: 0.374055, expected 7.30446</span></span>
<span class="line"><span>E0000 00:00:1746711234.779678 3123193 buffer_comparator.cc:145] Difference at 21: 0.216797, expected 8.26483</span></span>
<span class="line"><span>E0000 00:00:1746711234.779681 3123193 buffer_comparator.cc:145] Difference at 22: 0.731212, expected 10.8549</span></span>
<span class="line"><span>E0000 00:00:1746711234.779684 3123193 buffer_comparator.cc:145] Difference at 23: 0.700668, expected 7.87482</span></span>
<span class="line"><span>E0000 00:00:1746711234.779687 3123193 buffer_comparator.cc:145] Difference at 24: 0.5317, expected 9.78239</span></span>
<span class="line"><span>E0000 00:00:1746711234.779690 3123193 buffer_comparator.cc:145] Difference at 25: 0.24009, expected 11.3838</span></span>
<span class="line"><span>2025-05-08 13:33:54.779697: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746711234.781721 3123193 buffer_comparator.cc:145] Difference at 16: 0.196842, expected 9.68745</span></span>
<span class="line"><span>E0000 00:00:1746711234.781734 3123193 buffer_comparator.cc:145] Difference at 17: 0.688536, expected 10.1876</span></span>
<span class="line"><span>E0000 00:00:1746711234.781738 3123193 buffer_comparator.cc:145] Difference at 18: 0.927057, expected 8.84104</span></span>
<span class="line"><span>E0000 00:00:1746711234.781741 3123193 buffer_comparator.cc:145] Difference at 19: 0.579189, expected 10.0381</span></span>
<span class="line"><span>E0000 00:00:1746711234.781744 3123193 buffer_comparator.cc:145] Difference at 20: 0.374055, expected 7.30446</span></span>
<span class="line"><span>E0000 00:00:1746711234.781747 3123193 buffer_comparator.cc:145] Difference at 21: 0.216797, expected 8.26483</span></span>
<span class="line"><span>E0000 00:00:1746711234.781750 3123193 buffer_comparator.cc:145] Difference at 22: 0.731212, expected 10.8549</span></span>
<span class="line"><span>E0000 00:00:1746711234.781753 3123193 buffer_comparator.cc:145] Difference at 23: 0.700668, expected 7.87482</span></span>
<span class="line"><span>E0000 00:00:1746711234.781756 3123193 buffer_comparator.cc:145] Difference at 24: 0.5317, expected 9.78239</span></span>
<span class="line"><span>E0000 00:00:1746711234.781759 3123193 buffer_comparator.cc:145] Difference at 25: 0.24009, expected 11.3838</span></span>
<span class="line"><span>2025-05-08 13:33:54.781764: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746711234.783776 3123193 buffer_comparator.cc:145] Difference at 32: 0.766695, expected 9.13848</span></span>
<span class="line"><span>E0000 00:00:1746711234.783788 3123193 buffer_comparator.cc:145] Difference at 33: 1.02114, expected 7.0792</span></span>
<span class="line"><span>E0000 00:00:1746711234.783791 3123193 buffer_comparator.cc:145] Difference at 34: 0.0917029, expected 10.2155</span></span>
<span class="line"><span>E0000 00:00:1746711234.783795 3123193 buffer_comparator.cc:145] Difference at 35: 0.842239, expected 9.45231</span></span>
<span class="line"><span>E0000 00:00:1746711234.783798 3123193 buffer_comparator.cc:145] Difference at 36: 0.52163, expected 10.5298</span></span>
<span class="line"><span>E0000 00:00:1746711234.783801 3123193 buffer_comparator.cc:145] Difference at 37: 0.313266, expected 9.84508</span></span>
<span class="line"><span>E0000 00:00:1746711234.783804 3123193 buffer_comparator.cc:145] Difference at 38: 1.04173, expected 9.51338</span></span>
<span class="line"><span>E0000 00:00:1746711234.783807 3123193 buffer_comparator.cc:145] Difference at 39: 0.974961, expected 10.1471</span></span>
<span class="line"><span>E0000 00:00:1746711234.783810 3123193 buffer_comparator.cc:145] Difference at 40: 0.978602, expected 9.57115</span></span>
<span class="line"><span>E0000 00:00:1746711234.783813 3123193 buffer_comparator.cc:145] Difference at 41: 1.00507, expected 8.63119</span></span>
<span class="line"><span>2025-05-08 13:33:54.783818: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746711234.785864 3123193 buffer_comparator.cc:145] Difference at 64: 0.586121, expected 9.67458</span></span>
<span class="line"><span>E0000 00:00:1746711234.785876 3123193 buffer_comparator.cc:145] Difference at 65: 0.809946, expected 10.734</span></span>
<span class="line"><span>E0000 00:00:1746711234.785880 3123193 buffer_comparator.cc:145] Difference at 66: 0.423876, expected 10.6109</span></span>
<span class="line"><span>E0000 00:00:1746711234.785883 3123193 buffer_comparator.cc:145] Difference at 67: 0.65869, expected 8.23326</span></span>
<span class="line"><span>E0000 00:00:1746711234.785887 3123193 buffer_comparator.cc:145] Difference at 68: 1.08471, expected 8.19665</span></span>
<span class="line"><span>E0000 00:00:1746711234.785889 3123193 buffer_comparator.cc:145] Difference at 69: 0.449177, expected 9.30282</span></span>
<span class="line"><span>E0000 00:00:1746711234.785892 3123193 buffer_comparator.cc:145] Difference at 70: 0.988388, expected 8.16784</span></span>
<span class="line"><span>E0000 00:00:1746711234.785895 3123193 buffer_comparator.cc:145] Difference at 71: 0.849879, expected 9.34399</span></span>
<span class="line"><span>E0000 00:00:1746711234.785898 3123193 buffer_comparator.cc:145] Difference at 72: 1.03034, expected 9.36502</span></span>
<span class="line"><span>E0000 00:00:1746711234.785901 3123193 buffer_comparator.cc:145] Difference at 73: 0.892119, expected 8.82565</span></span>
<span class="line"><span>2025-05-08 13:33:54.785906: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746711234.788021 3123193 buffer_comparator.cc:145] Difference at 64: 0.586121, expected 9.67458</span></span>
<span class="line"><span>E0000 00:00:1746711234.788033 3123193 buffer_comparator.cc:145] Difference at 65: 0.809946, expected 10.734</span></span>
<span class="line"><span>E0000 00:00:1746711234.788037 3123193 buffer_comparator.cc:145] Difference at 66: 0.423876, expected 10.6109</span></span>
<span class="line"><span>E0000 00:00:1746711234.788040 3123193 buffer_comparator.cc:145] Difference at 67: 0.65869, expected 8.23326</span></span>
<span class="line"><span>E0000 00:00:1746711234.788043 3123193 buffer_comparator.cc:145] Difference at 68: 1.08471, expected 8.19665</span></span>
<span class="line"><span>E0000 00:00:1746711234.788046 3123193 buffer_comparator.cc:145] Difference at 69: 0.449177, expected 9.30282</span></span>
<span class="line"><span>E0000 00:00:1746711234.788049 3123193 buffer_comparator.cc:145] Difference at 70: 0.988388, expected 8.16784</span></span>
<span class="line"><span>E0000 00:00:1746711234.788052 3123193 buffer_comparator.cc:145] Difference at 71: 0.849879, expected 9.34399</span></span>
<span class="line"><span>E0000 00:00:1746711234.788055 3123193 buffer_comparator.cc:145] Difference at 72: 1.03034, expected 9.36502</span></span>
<span class="line"><span>E0000 00:00:1746711234.788058 3123193 buffer_comparator.cc:145] Difference at 73: 0.892119, expected 8.82565</span></span>
<span class="line"><span>2025-05-08 13:33:54.788062: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746711234.790162 3123193 buffer_comparator.cc:145] Difference at 64: 0.586121, expected 9.67458</span></span>
<span class="line"><span>E0000 00:00:1746711234.790187 3123193 buffer_comparator.cc:145] Difference at 65: 0.809946, expected 10.734</span></span>
<span class="line"><span>E0000 00:00:1746711234.790191 3123193 buffer_comparator.cc:145] Difference at 66: 0.423876, expected 10.6109</span></span>
<span class="line"><span>E0000 00:00:1746711234.790194 3123193 buffer_comparator.cc:145] Difference at 67: 0.65869, expected 8.23326</span></span>
<span class="line"><span>E0000 00:00:1746711234.790197 3123193 buffer_comparator.cc:145] Difference at 68: 1.08471, expected 8.19665</span></span>
<span class="line"><span>E0000 00:00:1746711234.790200 3123193 buffer_comparator.cc:145] Difference at 69: 0.449177, expected 9.30282</span></span>
<span class="line"><span>E0000 00:00:1746711234.790203 3123193 buffer_comparator.cc:145] Difference at 70: 0.988388, expected 8.16784</span></span>
<span class="line"><span>E0000 00:00:1746711234.790206 3123193 buffer_comparator.cc:145] Difference at 71: 0.849879, expected 9.34399</span></span>
<span class="line"><span>E0000 00:00:1746711234.790208 3123193 buffer_comparator.cc:145] Difference at 72: 1.03034, expected 9.36502</span></span>
<span class="line"><span>E0000 00:00:1746711234.790211 3123193 buffer_comparator.cc:145] Difference at 73: 0.892119, expected 8.82565</span></span>
<span class="line"><span>2025-05-08 13:33:54.790217: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746711234.792268 3123193 buffer_comparator.cc:145] Difference at 64: 0.586121, expected 9.67458</span></span>
<span class="line"><span>E0000 00:00:1746711234.792280 3123193 buffer_comparator.cc:145] Difference at 65: 0.809946, expected 10.734</span></span>
<span class="line"><span>E0000 00:00:1746711234.792289 3123193 buffer_comparator.cc:145] Difference at 66: 0.423876, expected 10.6109</span></span>
<span class="line"><span>E0000 00:00:1746711234.792292 3123193 buffer_comparator.cc:145] Difference at 67: 0.65869, expected 8.23326</span></span>
<span class="line"><span>E0000 00:00:1746711234.792295 3123193 buffer_comparator.cc:145] Difference at 68: 1.08471, expected 8.19665</span></span>
<span class="line"><span>E0000 00:00:1746711234.792298 3123193 buffer_comparator.cc:145] Difference at 69: 0.449177, expected 9.30282</span></span>
<span class="line"><span>E0000 00:00:1746711234.792301 3123193 buffer_comparator.cc:145] Difference at 70: 0.988388, expected 8.16784</span></span>
<span class="line"><span>E0000 00:00:1746711234.792304 3123193 buffer_comparator.cc:145] Difference at 71: 0.849879, expected 9.34399</span></span>
<span class="line"><span>E0000 00:00:1746711234.792307 3123193 buffer_comparator.cc:145] Difference at 72: 1.03034, expected 9.36502</span></span>
<span class="line"><span>E0000 00:00:1746711234.792310 3123193 buffer_comparator.cc:145] Difference at 73: 0.892119, expected 8.82565</span></span>
<span class="line"><span>2025-05-08 13:33:54.792315: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746711234.794352 3123193 buffer_comparator.cc:145] Difference at 64: 0.586121, expected 9.67458</span></span>
<span class="line"><span>E0000 00:00:1746711234.794365 3123193 buffer_comparator.cc:145] Difference at 65: 0.809946, expected 10.734</span></span>
<span class="line"><span>E0000 00:00:1746711234.794370 3123193 buffer_comparator.cc:145] Difference at 66: 0.423876, expected 10.6109</span></span>
<span class="line"><span>E0000 00:00:1746711234.794373 3123193 buffer_comparator.cc:145] Difference at 67: 0.65869, expected 8.23326</span></span>
<span class="line"><span>E0000 00:00:1746711234.794376 3123193 buffer_comparator.cc:145] Difference at 68: 1.08471, expected 8.19665</span></span>
<span class="line"><span>E0000 00:00:1746711234.794379 3123193 buffer_comparator.cc:145] Difference at 69: 0.449177, expected 9.30282</span></span>
<span class="line"><span>E0000 00:00:1746711234.794382 3123193 buffer_comparator.cc:145] Difference at 70: 0.988388, expected 8.16784</span></span>
<span class="line"><span>E0000 00:00:1746711234.794384 3123193 buffer_comparator.cc:145] Difference at 71: 0.849879, expected 9.34399</span></span>
<span class="line"><span>E0000 00:00:1746711234.794387 3123193 buffer_comparator.cc:145] Difference at 72: 1.03034, expected 9.36502</span></span>
<span class="line"><span>E0000 00:00:1746711234.794390 3123193 buffer_comparator.cc:145] Difference at 73: 0.892119, expected 8.82565</span></span>
<span class="line"><span>2025-05-08 13:33:54.794395: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746711234.796412 3123193 buffer_comparator.cc:145] Difference at 64: 0.586121, expected 9.67458</span></span>
<span class="line"><span>E0000 00:00:1746711234.796424 3123193 buffer_comparator.cc:145] Difference at 65: 0.809946, expected 10.734</span></span>
<span class="line"><span>E0000 00:00:1746711234.796428 3123193 buffer_comparator.cc:145] Difference at 66: 0.423876, expected 10.6109</span></span>
<span class="line"><span>E0000 00:00:1746711234.796431 3123193 buffer_comparator.cc:145] Difference at 67: 0.65869, expected 8.23326</span></span>
<span class="line"><span>E0000 00:00:1746711234.796434 3123193 buffer_comparator.cc:145] Difference at 68: 1.08471, expected 8.19665</span></span>
<span class="line"><span>E0000 00:00:1746711234.796437 3123193 buffer_comparator.cc:145] Difference at 69: 0.449177, expected 9.30282</span></span>
<span class="line"><span>E0000 00:00:1746711234.796440 3123193 buffer_comparator.cc:145] Difference at 70: 0.988388, expected 8.16784</span></span>
<span class="line"><span>E0000 00:00:1746711234.796443 3123193 buffer_comparator.cc:145] Difference at 71: 0.849879, expected 9.34399</span></span>
<span class="line"><span>E0000 00:00:1746711234.796446 3123193 buffer_comparator.cc:145] Difference at 72: 1.03034, expected 9.36502</span></span>
<span class="line"><span>E0000 00:00:1746711234.796449 3123193 buffer_comparator.cc:145] Difference at 73: 0.892119, expected 8.82565</span></span>
<span class="line"><span>2025-05-08 13:33:54.796454: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746711234.801897 3123193 buffer_comparator.cc:145] Difference at 16: 9.68831, expected 34.2325</span></span>
<span class="line"><span>E0000 00:00:1746711234.801921 3123193 buffer_comparator.cc:145] Difference at 17: 10.1886, expected 32.4845</span></span>
<span class="line"><span>E0000 00:00:1746711234.801925 3123193 buffer_comparator.cc:145] Difference at 18: 8.84087, expected 35.8503</span></span>
<span class="line"><span>E0000 00:00:1746711234.801928 3123193 buffer_comparator.cc:145] Difference at 19: 10.0385, expected 38.0823</span></span>
<span class="line"><span>E0000 00:00:1746711234.801932 3123193 buffer_comparator.cc:145] Difference at 20: 7.30459, expected 32.6811</span></span>
<span class="line"><span>E0000 00:00:1746711234.801935 3123193 buffer_comparator.cc:145] Difference at 21: 8.26478, expected 37.818</span></span>
<span class="line"><span>E0000 00:00:1746711234.801938 3123193 buffer_comparator.cc:145] Difference at 22: 10.8556, expected 35.4896</span></span>
<span class="line"><span>E0000 00:00:1746711234.801941 3123193 buffer_comparator.cc:145] Difference at 23: 7.87467, expected 35.057</span></span>
<span class="line"><span>E0000 00:00:1746711234.801944 3123193 buffer_comparator.cc:145] Difference at 24: 9.78306, expected 37.6513</span></span>
<span class="line"><span>E0000 00:00:1746711234.801947 3123193 buffer_comparator.cc:145] Difference at 25: 11.3832, expected 36.0917</span></span>
<span class="line"><span>2025-05-08 13:33:54.801954: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746711234.803961 3123193 buffer_comparator.cc:145] Difference at 16: 9.68831, expected 34.2325</span></span>
<span class="line"><span>E0000 00:00:1746711234.803973 3123193 buffer_comparator.cc:145] Difference at 17: 10.1886, expected 32.4845</span></span>
<span class="line"><span>E0000 00:00:1746711234.803977 3123193 buffer_comparator.cc:145] Difference at 18: 8.84087, expected 35.8503</span></span>
<span class="line"><span>E0000 00:00:1746711234.803982 3123193 buffer_comparator.cc:145] Difference at 19: 10.0385, expected 38.0823</span></span>
<span class="line"><span>E0000 00:00:1746711234.803985 3123193 buffer_comparator.cc:145] Difference at 20: 7.30459, expected 32.6811</span></span>
<span class="line"><span>E0000 00:00:1746711234.803988 3123193 buffer_comparator.cc:145] Difference at 21: 8.26478, expected 37.818</span></span>
<span class="line"><span>E0000 00:00:1746711234.803991 3123193 buffer_comparator.cc:145] Difference at 22: 10.8556, expected 35.4896</span></span>
<span class="line"><span>E0000 00:00:1746711234.803995 3123193 buffer_comparator.cc:145] Difference at 23: 7.87467, expected 35.057</span></span>
<span class="line"><span>E0000 00:00:1746711234.803998 3123193 buffer_comparator.cc:145] Difference at 24: 9.78306, expected 37.6513</span></span>
<span class="line"><span>E0000 00:00:1746711234.804001 3123193 buffer_comparator.cc:145] Difference at 25: 11.3832, expected 36.0917</span></span>
<span class="line"><span>2025-05-08 13:33:54.804005: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746711234.806001 3123193 buffer_comparator.cc:145] Difference at 16: 9.68831, expected 34.2325</span></span>
<span class="line"><span>E0000 00:00:1746711234.806013 3123193 buffer_comparator.cc:145] Difference at 17: 10.1886, expected 32.4845</span></span>
<span class="line"><span>E0000 00:00:1746711234.806017 3123193 buffer_comparator.cc:145] Difference at 18: 8.84087, expected 35.8503</span></span>
<span class="line"><span>E0000 00:00:1746711234.806021 3123193 buffer_comparator.cc:145] Difference at 19: 10.0385, expected 38.0823</span></span>
<span class="line"><span>E0000 00:00:1746711234.806024 3123193 buffer_comparator.cc:145] Difference at 20: 7.30459, expected 32.6811</span></span>
<span class="line"><span>E0000 00:00:1746711234.806027 3123193 buffer_comparator.cc:145] Difference at 21: 8.26478, expected 37.818</span></span>
<span class="line"><span>E0000 00:00:1746711234.806030 3123193 buffer_comparator.cc:145] Difference at 22: 10.8556, expected 35.4896</span></span>
<span class="line"><span>E0000 00:00:1746711234.806033 3123193 buffer_comparator.cc:145] Difference at 23: 7.87467, expected 35.057</span></span>
<span class="line"><span>E0000 00:00:1746711234.806036 3123193 buffer_comparator.cc:145] Difference at 24: 9.78306, expected 37.6513</span></span>
<span class="line"><span>E0000 00:00:1746711234.806039 3123193 buffer_comparator.cc:145] Difference at 25: 11.3832, expected 36.0917</span></span>
<span class="line"><span>2025-05-08 13:33:54.806044: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746711234.808040 3123193 buffer_comparator.cc:145] Difference at 16: 9.68831, expected 34.2325</span></span>
<span class="line"><span>E0000 00:00:1746711234.808052 3123193 buffer_comparator.cc:145] Difference at 17: 10.1886, expected 32.4845</span></span>
<span class="line"><span>E0000 00:00:1746711234.808057 3123193 buffer_comparator.cc:145] Difference at 18: 8.84087, expected 35.8503</span></span>
<span class="line"><span>E0000 00:00:1746711234.808060 3123193 buffer_comparator.cc:145] Difference at 19: 10.0385, expected 38.0823</span></span>
<span class="line"><span>E0000 00:00:1746711234.808063 3123193 buffer_comparator.cc:145] Difference at 20: 7.30459, expected 32.6811</span></span>
<span class="line"><span>E0000 00:00:1746711234.808066 3123193 buffer_comparator.cc:145] Difference at 21: 8.26478, expected 37.818</span></span>
<span class="line"><span>E0000 00:00:1746711234.808069 3123193 buffer_comparator.cc:145] Difference at 22: 10.8556, expected 35.4896</span></span>
<span class="line"><span>E0000 00:00:1746711234.808072 3123193 buffer_comparator.cc:145] Difference at 23: 7.87467, expected 35.057</span></span>
<span class="line"><span>E0000 00:00:1746711234.808075 3123193 buffer_comparator.cc:145] Difference at 24: 9.78306, expected 37.6513</span></span>
<span class="line"><span>E0000 00:00:1746711234.808078 3123193 buffer_comparator.cc:145] Difference at 25: 11.3832, expected 36.0917</span></span>
<span class="line"><span>2025-05-08 13:33:54.808083: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746711234.810090 3123193 buffer_comparator.cc:145] Difference at 16: 9.68831, expected 34.2325</span></span>
<span class="line"><span>E0000 00:00:1746711234.810102 3123193 buffer_comparator.cc:145] Difference at 17: 10.1886, expected 32.4845</span></span>
<span class="line"><span>E0000 00:00:1746711234.810106 3123193 buffer_comparator.cc:145] Difference at 18: 8.84087, expected 35.8503</span></span>
<span class="line"><span>E0000 00:00:1746711234.810110 3123193 buffer_comparator.cc:145] Difference at 19: 10.0385, expected 38.0823</span></span>
<span class="line"><span>E0000 00:00:1746711234.810113 3123193 buffer_comparator.cc:145] Difference at 20: 7.30459, expected 32.6811</span></span>
<span class="line"><span>E0000 00:00:1746711234.810117 3123193 buffer_comparator.cc:145] Difference at 21: 8.26478, expected 37.818</span></span>
<span class="line"><span>E0000 00:00:1746711234.810120 3123193 buffer_comparator.cc:145] Difference at 22: 10.8556, expected 35.4896</span></span>
<span class="line"><span>E0000 00:00:1746711234.810123 3123193 buffer_comparator.cc:145] Difference at 23: 7.87467, expected 35.057</span></span>
<span class="line"><span>E0000 00:00:1746711234.810126 3123193 buffer_comparator.cc:145] Difference at 24: 9.78306, expected 37.6513</span></span>
<span class="line"><span>E0000 00:00:1746711234.810129 3123193 buffer_comparator.cc:145] Difference at 25: 11.3832, expected 36.0917</span></span>
<span class="line"><span>2025-05-08 13:33:54.810134: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746711234.830726 3123193 buffer_comparator.cc:145] Difference at 16: -nan, expected 34.687</span></span>
<span class="line"><span>E0000 00:00:1746711234.830758 3123193 buffer_comparator.cc:145] Difference at 17: -nan, expected 32.6585</span></span>
<span class="line"><span>E0000 00:00:1746711234.830765 3123193 buffer_comparator.cc:145] Difference at 18: -nan, expected 37.2083</span></span>
<span class="line"><span>E0000 00:00:1746711234.830768 3123193 buffer_comparator.cc:145] Difference at 19: -nan, expected 32.2063</span></span>
<span class="line"><span>E0000 00:00:1746711234.830771 3123193 buffer_comparator.cc:145] Difference at 20: -nan, expected 33.4727</span></span>
<span class="line"><span>E0000 00:00:1746711234.830774 3123193 buffer_comparator.cc:145] Difference at 21: -nan, expected 33.0033</span></span>
<span class="line"><span>E0000 00:00:1746711234.830777 3123193 buffer_comparator.cc:145] Difference at 22: -nan, expected 31.6193</span></span>
<span class="line"><span>E0000 00:00:1746711234.830780 3123193 buffer_comparator.cc:145] Difference at 23: -nan, expected 32.1492</span></span>
<span class="line"><span>E0000 00:00:1746711234.830783 3123193 buffer_comparator.cc:145] Difference at 24: -nan, expected 32.5713</span></span>
<span class="line"><span>E0000 00:00:1746711234.830785 3123193 buffer_comparator.cc:145] Difference at 25: -nan, expected 36.4575</span></span>
<span class="line"><span>2025-05-08 13:33:54.830793: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746711234.832802 3123193 buffer_comparator.cc:145] Difference at 16: -nan, expected 34.687</span></span>
<span class="line"><span>E0000 00:00:1746711234.832814 3123193 buffer_comparator.cc:145] Difference at 17: -nan, expected 32.6585</span></span>
<span class="line"><span>E0000 00:00:1746711234.832818 3123193 buffer_comparator.cc:145] Difference at 18: -nan, expected 37.2083</span></span>
<span class="line"><span>E0000 00:00:1746711234.832821 3123193 buffer_comparator.cc:145] Difference at 19: -nan, expected 32.2063</span></span>
<span class="line"><span>E0000 00:00:1746711234.832824 3123193 buffer_comparator.cc:145] Difference at 20: -nan, expected 33.4727</span></span>
<span class="line"><span>E0000 00:00:1746711234.832827 3123193 buffer_comparator.cc:145] Difference at 21: -nan, expected 33.0033</span></span>
<span class="line"><span>E0000 00:00:1746711234.832830 3123193 buffer_comparator.cc:145] Difference at 22: -nan, expected 31.6193</span></span>
<span class="line"><span>E0000 00:00:1746711234.832832 3123193 buffer_comparator.cc:145] Difference at 23: -nan, expected 32.1492</span></span>
<span class="line"><span>E0000 00:00:1746711234.832835 3123193 buffer_comparator.cc:145] Difference at 24: -nan, expected 32.5713</span></span>
<span class="line"><span>E0000 00:00:1746711234.832838 3123193 buffer_comparator.cc:145] Difference at 25: -nan, expected 36.4575</span></span>
<span class="line"><span>2025-05-08 13:33:54.832843: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746711234.834857 3123193 buffer_comparator.cc:145] Difference at 16: -nan, expected 34.687</span></span>
<span class="line"><span>E0000 00:00:1746711234.834869 3123193 buffer_comparator.cc:145] Difference at 17: -nan, expected 32.6585</span></span>
<span class="line"><span>E0000 00:00:1746711234.834873 3123193 buffer_comparator.cc:145] Difference at 18: -nan, expected 37.2083</span></span>
<span class="line"><span>E0000 00:00:1746711234.834876 3123193 buffer_comparator.cc:145] Difference at 19: -nan, expected 32.2063</span></span>
<span class="line"><span>E0000 00:00:1746711234.834879 3123193 buffer_comparator.cc:145] Difference at 20: -nan, expected 33.4727</span></span>
<span class="line"><span>E0000 00:00:1746711234.834882 3123193 buffer_comparator.cc:145] Difference at 21: -nan, expected 33.0033</span></span>
<span class="line"><span>E0000 00:00:1746711234.834885 3123193 buffer_comparator.cc:145] Difference at 22: -nan, expected 31.6193</span></span>
<span class="line"><span>E0000 00:00:1746711234.834888 3123193 buffer_comparator.cc:145] Difference at 23: -nan, expected 32.1492</span></span>
<span class="line"><span>E0000 00:00:1746711234.834892 3123193 buffer_comparator.cc:145] Difference at 24: -nan, expected 32.5713</span></span>
<span class="line"><span>E0000 00:00:1746711234.834895 3123193 buffer_comparator.cc:145] Difference at 25: -nan, expected 36.4575</span></span>
<span class="line"><span>2025-05-08 13:33:54.834900: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746711234.836895 3123193 buffer_comparator.cc:145] Difference at 16: -nan, expected 34.687</span></span>
<span class="line"><span>E0000 00:00:1746711234.836907 3123193 buffer_comparator.cc:145] Difference at 17: -nan, expected 32.6585</span></span>
<span class="line"><span>E0000 00:00:1746711234.836910 3123193 buffer_comparator.cc:145] Difference at 18: -nan, expected 37.2083</span></span>
<span class="line"><span>E0000 00:00:1746711234.836913 3123193 buffer_comparator.cc:145] Difference at 19: -nan, expected 32.2063</span></span>
<span class="line"><span>E0000 00:00:1746711234.836916 3123193 buffer_comparator.cc:145] Difference at 20: -nan, expected 33.4727</span></span>
<span class="line"><span>E0000 00:00:1746711234.836919 3123193 buffer_comparator.cc:145] Difference at 21: -nan, expected 33.0033</span></span>
<span class="line"><span>E0000 00:00:1746711234.836922 3123193 buffer_comparator.cc:145] Difference at 22: -nan, expected 31.6193</span></span>
<span class="line"><span>E0000 00:00:1746711234.836925 3123193 buffer_comparator.cc:145] Difference at 23: -nan, expected 32.1492</span></span>
<span class="line"><span>E0000 00:00:1746711234.836927 3123193 buffer_comparator.cc:145] Difference at 24: -nan, expected 32.5713</span></span>
<span class="line"><span>E0000 00:00:1746711234.836930 3123193 buffer_comparator.cc:145] Difference at 25: -nan, expected 36.4575</span></span>
<span class="line"><span>2025-05-08 13:33:54.836935: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1746711234.838930 3123193 buffer_comparator.cc:145] Difference at 16: -nan, expected 34.687</span></span>
<span class="line"><span>E0000 00:00:1746711234.838944 3123193 buffer_comparator.cc:145] Difference at 17: -nan, expected 32.6585</span></span>
<span class="line"><span>E0000 00:00:1746711234.838948 3123193 buffer_comparator.cc:145] Difference at 18: -nan, expected 37.2083</span></span>
<span class="line"><span>E0000 00:00:1746711234.838952 3123193 buffer_comparator.cc:145] Difference at 19: -nan, expected 32.2063</span></span>
<span class="line"><span>E0000 00:00:1746711234.838955 3123193 buffer_comparator.cc:145] Difference at 20: -nan, expected 33.4727</span></span>
<span class="line"><span>E0000 00:00:1746711234.838958 3123193 buffer_comparator.cc:145] Difference at 21: -nan, expected 33.0033</span></span>
<span class="line"><span>E0000 00:00:1746711234.838961 3123193 buffer_comparator.cc:145] Difference at 22: -nan, expected 31.6193</span></span>
<span class="line"><span>E0000 00:00:1746711234.838963 3123193 buffer_comparator.cc:145] Difference at 23: -nan, expected 32.1492</span></span>
<span class="line"><span>E0000 00:00:1746711234.838966 3123193 buffer_comparator.cc:145] Difference at 24: -nan, expected 32.5713</span></span>
<span class="line"><span>E0000 00:00:1746711234.838969 3123193 buffer_comparator.cc:145] Difference at 25: -nan, expected 36.4575</span></span>
<span class="line"><span>2025-05-08 13:33:54.838974: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1179] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.77102</span></span>
<span class="line"><span>Validation:	Loss 0.71518	Accuracy 0.49219</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.66183</span></span>
<span class="line"><span>Validation:	Loss 0.58588	Accuracy 0.49219</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.52172</span></span>
<span class="line"><span>Validation:	Loss 0.43412	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.37535</span></span>
<span class="line"><span>Validation:	Loss 0.30415	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.26135</span></span>
<span class="line"><span>Validation:	Loss 0.21227	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.18548</span></span>
<span class="line"><span>Validation:	Loss 0.15504	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.13831</span></span>
<span class="line"><span>Validation:	Loss 0.11817	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.10739</span></span>
<span class="line"><span>Validation:	Loss 0.09384	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.08644</span></span>
<span class="line"><span>Validation:	Loss 0.07665	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.07124</span></span>
<span class="line"><span>Validation:	Loss 0.06392	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.05986</span></span>
<span class="line"><span>Validation:	Loss 0.05431	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.05111</span></span>
<span class="line"><span>Validation:	Loss 0.04649	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.04373</span></span>
<span class="line"><span>Validation:	Loss 0.03968	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.03703</span></span>
<span class="line"><span>Validation:	Loss 0.03322	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.03061</span></span>
<span class="line"><span>Validation:	Loss 0.02685	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.02452</span></span>
<span class="line"><span>Validation:	Loss 0.02131	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01978</span></span>
<span class="line"><span>Validation:	Loss 0.01757	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01659</span></span>
<span class="line"><span>Validation:	Loss 0.01510	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01432</span></span>
<span class="line"><span>Validation:	Loss 0.01322	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01265</span></span>
<span class="line"><span>Validation:	Loss 0.01163	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01110</span></span>
<span class="line"><span>Validation:	Loss 0.01018	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00971</span></span>
<span class="line"><span>Validation:	Loss 0.00878	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00831</span></span>
<span class="line"><span>Validation:	Loss 0.00747	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00706</span></span>
<span class="line"><span>Validation:	Loss 0.00638	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00610</span></span>
<span class="line"><span>Validation:	Loss 0.00559	Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>┌ Warning: \`replicate\` doesn&#39;t work for \`TaskLocalRNG\`. Returning the same \`TaskLocalRNG\`.</span></span>
<span class="line"><span>└ @ LuxCore /var/lib/buildkite-agent/builds/gpuci-10/julialang/lux-dot-jl/lib/LuxCore/src/LuxCore.jl:18</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.94544</span></span>
<span class="line"><span>Validation:	Loss 0.69613	Accuracy 0.56250</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.61169</span></span>
<span class="line"><span>Validation:	Loss 0.50845	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.45723</span></span>
<span class="line"><span>Validation:	Loss 0.43266	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.38456</span></span>
<span class="line"><span>Validation:	Loss 0.38257	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.33964</span></span>
<span class="line"><span>Validation:	Loss 0.33890	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.29663</span></span>
<span class="line"><span>Validation:	Loss 0.29796	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.25725</span></span>
<span class="line"><span>Validation:	Loss 0.26062	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.22534</span></span>
<span class="line"><span>Validation:	Loss 0.22709	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.19821</span></span>
<span class="line"><span>Validation:	Loss 0.19755	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.17055</span></span>
<span class="line"><span>Validation:	Loss 0.17127	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.14845</span></span>
<span class="line"><span>Validation:	Loss 0.14720	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.12852</span></span>
<span class="line"><span>Validation:	Loss 0.12489	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.10823</span></span>
<span class="line"><span>Validation:	Loss 0.10390	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.08985</span></span>
<span class="line"><span>Validation:	Loss 0.08369	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.07280</span></span>
<span class="line"><span>Validation:	Loss 0.06836	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.06158</span></span>
<span class="line"><span>Validation:	Loss 0.05989	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.05491</span></span>
<span class="line"><span>Validation:	Loss 0.05404	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.04983</span></span>
<span class="line"><span>Validation:	Loss 0.04915	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.04534</span></span>
<span class="line"><span>Validation:	Loss 0.04498	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.04150</span></span>
<span class="line"><span>Validation:	Loss 0.04140	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.03821</span></span>
<span class="line"><span>Validation:	Loss 0.03831	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.03541</span></span>
<span class="line"><span>Validation:	Loss 0.03558	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.03282</span></span>
<span class="line"><span>Validation:	Loss 0.03319	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.03068</span></span>
<span class="line"><span>Validation:	Loss 0.03105	Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.02865</span></span>
<span class="line"><span>Validation:	Loss 0.02915	Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model struct and only save the parameters and states.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Julia Version 1.11.5</span></span>
<span class="line"><span>Commit 760b2e5b739 (2025-04-14 06:53 UTC)</span></span>
<span class="line"><span>Build Info:</span></span>
<span class="line"><span>  Official https://julialang.org/ release</span></span>
<span class="line"><span>Platform Info:</span></span>
<span class="line"><span>  OS: Linux (x86_64-linux-gnu)</span></span>
<span class="line"><span>  CPU: 48 × AMD EPYC 7402 24-Core Processor</span></span>
<span class="line"><span>  WORD_SIZE: 64</span></span>
<span class="line"><span>  LLVM: libLLVM-16.0.6 (ORCJIT, znver2)</span></span>
<span class="line"><span>Threads: 48 default, 0 interactive, 24 GC (on 2 virtual cores)</span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>  JULIA_CPU_THREADS = 2</span></span>
<span class="line"><span>  LD_LIBRARY_PATH = /usr/local/nvidia/lib:/usr/local/nvidia/lib64</span></span>
<span class="line"><span>  JULIA_PKG_SERVER = </span></span>
<span class="line"><span>  JULIA_NUM_THREADS = 48</span></span>
<span class="line"><span>  JULIA_CUDA_HARD_MEMORY_LIMIT = 100%</span></span>
<span class="line"><span>  JULIA_PKG_PRECOMPILE_AUTO = 0</span></span>
<span class="line"><span>  JULIA_DEBUG = Literate</span></span>
<span class="line"><span>  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,46)]))}const d=a(p,[["render",t]]);export{k as __pageData,d as default};
