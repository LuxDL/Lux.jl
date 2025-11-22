import{_ as s,c as a,o as n,a3 as i}from"./chunks/framework.BrB0cdsk.js";const y=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),p={name:"tutorials/beginner/3_SimpleRNN.md"},l=i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, AMDGPU, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">      Statistics</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractExplicitContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/v0.5.58/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">:</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">       Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">AbstractExplicitContainerLayer{(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/v0.5.58/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/v0.5.58/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Main.var&quot;##225&quot;.SpiralClassifier</span></span></code></pre></div><p>We can use default Lux blocks – <code>Recurrence(LSTMCell(in_dims =&gt; hidden_dims)</code> – instead of defining the following. But let&#39;s still do it for the sake of it.</p><p>Now we need to define the behavior of the Classifier when it is invoked.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)(</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/v0.5.58/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            (_, loss, _, train_state) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Experimental</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">single_train_step!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                AutoZygote</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), lossfn, (x, y), train_state)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Epoch [%3d]: Loss %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch loss</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Validate the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> val_loader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.55985</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50974</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47781</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.44590</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.41706</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.40720</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.39376</span></span>
<span class="line"><span>Validation: Loss 0.35978 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37656 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.35936</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.35656</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.32952</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.30833</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.30292</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.29106</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.27066</span></span>
<span class="line"><span>Validation: Loss 0.25268 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.26268 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.25933</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24381</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22898</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22053</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21108</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19478</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.18931</span></span>
<span class="line"><span>Validation: Loss 0.17711 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.18173 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17945</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17065</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16134</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15489</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14825</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13843</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13151</span></span>
<span class="line"><span>Validation: Loss 0.12629 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12911 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12698</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12108</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11533</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10974</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10655</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10291</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09732</span></span>
<span class="line"><span>Validation: Loss 0.09116 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09432 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09233</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08898</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08439</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08112</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07658</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07458</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07191</span></span>
<span class="line"><span>Validation: Loss 0.06678 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.07011 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06761</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06536</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06148</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05967</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05794</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05563</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05889</span></span>
<span class="line"><span>Validation: Loss 0.04945 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05278 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05174</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04895</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04643</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04448</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04276</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04180</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04055</span></span>
<span class="line"><span>Validation: Loss 0.03699 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04003 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03787</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03604</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03386</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03422</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03571</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03171</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.02835</span></span>
<span class="line"><span>Validation: Loss 0.02806 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03089 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02892</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02775</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02652</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02672</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02500</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02623</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02372</span></span>
<span class="line"><span>Validation: Loss 0.02194 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02451 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02296</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02229</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02124</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02072</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02085</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02044</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01869</span></span>
<span class="line"><span>Validation: Loss 0.01780 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02006 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01767</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01839</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01836</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01712</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01700</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01693</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01703</span></span>
<span class="line"><span>Validation: Loss 0.01494 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01691 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01524</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01523</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01567</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01494</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01532</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01370</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01271</span></span>
<span class="line"><span>Validation: Loss 0.01290 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01462 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01405</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01409</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01240</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01302</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01281</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01192</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01183</span></span>
<span class="line"><span>Validation: Loss 0.01136 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01291 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01210</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01104</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01142</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01213</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01149</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01118</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01062</span></span>
<span class="line"><span>Validation: Loss 0.01017 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01157 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01075</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01134</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00975</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01081</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00993</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00992</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00926</span></span>
<span class="line"><span>Validation: Loss 0.00920 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01048 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00952</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00940</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00891</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00938</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01008</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00914</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00970</span></span>
<span class="line"><span>Validation: Loss 0.00840 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00959 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00892</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00879</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00868</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00892</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00804</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00836</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00878</span></span>
<span class="line"><span>Validation: Loss 0.00772 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00881 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00899</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00761</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00825</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00789</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00764</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00735</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00764</span></span>
<span class="line"><span>Validation: Loss 0.00712 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00815 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00747</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00690</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00761</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00763</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00708</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00767</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00625</span></span>
<span class="line"><span>Validation: Loss 0.00661 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00757 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00716</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00687</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00746</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00626</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00672</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00662</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00659</span></span>
<span class="line"><span>Validation: Loss 0.00617 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00706 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00674</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00686</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00635</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00630</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00618</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00585</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00653</span></span>
<span class="line"><span>Validation: Loss 0.00576 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00661 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00593</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00611</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00570</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00618</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00611</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00587</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00579</span></span>
<span class="line"><span>Validation: Loss 0.00541 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00621 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00551</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00572</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00567</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00563</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00595</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00538</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00495</span></span>
<span class="line"><span>Validation: Loss 0.00509 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00584 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00540</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00538</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00519</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00530</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00554</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00525</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00407</span></span>
<span class="line"><span>Validation: Loss 0.00480 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00552 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.56125</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50935</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47948</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.45799</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.42040</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.40756</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.38547</span></span>
<span class="line"><span>Validation: Loss 0.37449 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.38109 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36658</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.35173</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.34233</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.31299</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.30644</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28310</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28431</span></span>
<span class="line"><span>Validation: Loss 0.26251 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.26673 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.25911</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24766</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22976</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22266</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21165</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.20401</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.18740</span></span>
<span class="line"><span>Validation: Loss 0.18312 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.18534 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.18302</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17118</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16510</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15768</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15001</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14093</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13755</span></span>
<span class="line"><span>Validation: Loss 0.13089 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.13235 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.13198</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12463</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11650</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11498</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10799</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10126</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09906</span></span>
<span class="line"><span>Validation: Loss 0.09553 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09680 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09473</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08966</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08566</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08498</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07950</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07543</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07427</span></span>
<span class="line"><span>Validation: Loss 0.07083 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.07211 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06929</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06585</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06392</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05906</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06100</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05882</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05640</span></span>
<span class="line"><span>Validation: Loss 0.05311 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05435 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05205</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05125</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04932</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04550</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04200</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04390</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.03822</span></span>
<span class="line"><span>Validation: Loss 0.04010 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04122 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03860</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03762</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03691</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03615</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03282</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03197</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03081</span></span>
<span class="line"><span>Validation: Loss 0.03074 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03177 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02832</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02929</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02974</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02647</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02616</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02425</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02635</span></span>
<span class="line"><span>Validation: Loss 0.02422 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02514 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02220</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02250</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02208</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02293</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02040</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02039</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02095</span></span>
<span class="line"><span>Validation: Loss 0.01971 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02050 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01956</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01736</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01927</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01734</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01668</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01707</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01754</span></span>
<span class="line"><span>Validation: Loss 0.01654 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01723 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01553</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01601</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01580</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01461</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01453</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01490</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01300</span></span>
<span class="line"><span>Validation: Loss 0.01426 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01487 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01434</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01357</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01389</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01347</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01209</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01143</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01407</span></span>
<span class="line"><span>Validation: Loss 0.01256 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01309 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01239</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01197</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01196</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01222</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01098</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01099</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.00949</span></span>
<span class="line"><span>Validation: Loss 0.01123 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01171 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01084</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01037</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01075</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01112</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00981</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00994</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01034</span></span>
<span class="line"><span>Validation: Loss 0.01016 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01060 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00977</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00929</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00916</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00962</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00987</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00935</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00915</span></span>
<span class="line"><span>Validation: Loss 0.00928 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00969 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00918</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00905</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00903</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00834</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00839</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00831</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00810</span></span>
<span class="line"><span>Validation: Loss 0.00852 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00890 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00834</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00810</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00832</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00767</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00758</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00816</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00731</span></span>
<span class="line"><span>Validation: Loss 0.00787 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00823 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00788</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00764</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00738</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00686</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00754</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00739</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00634</span></span>
<span class="line"><span>Validation: Loss 0.00731 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00765 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00703</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00684</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00722</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00664</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00685</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00674</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00672</span></span>
<span class="line"><span>Validation: Loss 0.00681 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00713 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00665</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00621</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00658</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00638</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00667</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00609</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00627</span></span>
<span class="line"><span>Validation: Loss 0.00638 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00668 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00618</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00617</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00596</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00595</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00618</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00567</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00591</span></span>
<span class="line"><span>Validation: Loss 0.00598 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00626 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00565</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00567</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00546</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00575</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00543</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00590</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00569</span></span>
<span class="line"><span>Validation: Loss 0.00563 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00590 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00571</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00510</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00557</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00532</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00518</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00496</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00557</span></span>
<span class="line"><span>Validation: Loss 0.00531 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00556 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {compress </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">} ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
<span class="line"><span> :ps_trained</span></span>
<span class="line"><span> :st_trained</span></span></code></pre></div><h2 id="appendix" tabindex="-1">Appendix <a class="header-anchor" href="#appendix" aria-label="Permalink to &quot;Appendix&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">InteractiveUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxDeviceUtils)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(CUDA) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&amp;&amp;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> LuxDeviceUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxCUDADevice)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        CUDA</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(AMDGPU) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&amp;&amp;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> LuxDeviceUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxAMDGPUDevice)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        AMDGPU</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Julia Version 1.10.4</span></span>
<span class="line"><span>Commit 48d4fd48430 (2024-06-04 10:41 UTC)</span></span>
<span class="line"><span>Build Info:</span></span>
<span class="line"><span>  Official https://julialang.org/ release</span></span>
<span class="line"><span>Platform Info:</span></span>
<span class="line"><span>  OS: Linux (x86_64-linux-gnu)</span></span>
<span class="line"><span>  CPU: 48 × AMD EPYC 7402 24-Core Processor</span></span>
<span class="line"><span>  WORD_SIZE: 64</span></span>
<span class="line"><span>  LIBM: libopenlibm</span></span>
<span class="line"><span>  LLVM: libLLVM-15.0.7 (ORCJIT, znver2)</span></span>
<span class="line"><span>Threads: 4 default, 0 interactive, 2 GC (on 2 virtual cores)</span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>  JULIA_CPU_THREADS = 2</span></span>
<span class="line"><span>  JULIA_AMDGPU_LOGGING_ENABLED = true</span></span>
<span class="line"><span>  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6</span></span>
<span class="line"><span>  LD_LIBRARY_PATH = /usr/local/nvidia/lib:/usr/local/nvidia/lib64</span></span>
<span class="line"><span>  JULIA_PKG_SERVER = </span></span>
<span class="line"><span>  JULIA_NUM_THREADS = 4</span></span>
<span class="line"><span>  JULIA_CUDA_HARD_MEMORY_LIMIT = 100%</span></span>
<span class="line"><span>  JULIA_PKG_PRECOMPILE_AUTO = 0</span></span>
<span class="line"><span>  JULIA_DEBUG = Literate</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA runtime 12.5, artifact installation</span></span>
<span class="line"><span>CUDA driver 12.5</span></span>
<span class="line"><span>NVIDIA driver 555.42.2</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA libraries: </span></span>
<span class="line"><span>- CUBLAS: 12.5.2</span></span>
<span class="line"><span>- CURAND: 10.3.6</span></span>
<span class="line"><span>- CUFFT: 11.2.3</span></span>
<span class="line"><span>- CUSOLVER: 11.6.2</span></span>
<span class="line"><span>- CUSPARSE: 12.4.1</span></span>
<span class="line"><span>- CUPTI: 23.0.0</span></span>
<span class="line"><span>- NVML: 12.0.0+555.42.2</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Julia packages: </span></span>
<span class="line"><span>- CUDA: 5.4.2</span></span>
<span class="line"><span>- CUDA_Driver_jll: 0.9.0+0</span></span>
<span class="line"><span>- CUDA_Runtime_jll: 0.14.0+1</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Toolchain:</span></span>
<span class="line"><span>- Julia: 1.10.4</span></span>
<span class="line"><span>- LLVM: 15.0.7</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>- JULIA_CUDA_HARD_MEMORY_LIMIT: 100%</span></span>
<span class="line"><span></span></span>
<span class="line"><span>1 device:</span></span>
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.453 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45),h=[l];function e(t,c,k,o,E,r){return n(),a("div",null,h)}const g=s(p,[["render",e]]);export{y as __pageData,g as default};
