import{_ as s,c as a,o as n,a3 as i}from"./chunks/framework.B5P1ePk0.js";const y=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),p={name:"tutorials/beginner/3_SimpleRNN.md"},l=i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxAMDGPU, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">      Statistics</span></span></code></pre></div><h2 id="Dataset" tabindex="-1">Dataset <a class="header-anchor" href="#Dataset" aria-label="Permalink to &quot;Dataset {#Dataset}&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractExplicitContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/v0.5.52/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">:</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">       Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">AbstractExplicitContainerLayer{(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/v0.5.52/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/v0.5.52/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/v0.5.52/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>SpiralClassifierCompact (generic function with 1 method)</span></span></code></pre></div><h2 id="Defining-Accuracy,-Loss-and-Optimiser" tabindex="-1">Defining Accuracy, Loss and Optimiser <a class="header-anchor" href="#Defining-Accuracy,-Loss-and-Optimiser" aria-label="Permalink to &quot;Defining Accuracy, Loss and Optimiser {#Defining-Accuracy,-Loss-and-Optimiser}&quot;">​</a></h2><p>Now let&#39;s define the binarycrossentropy loss. Typically it is recommended to use <code>logitbinarycrossentropy</code> since it is more numerically stable, but for the sake of simplicity we will use <code>binarycrossentropy</code>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> xlogy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, y)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.56097</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51558</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48168</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.44571</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.42640</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.41915</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.38075</span></span>
<span class="line"><span>Validation: Loss 0.37506 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36210 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37032</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.35682</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.34059</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.32744</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.29744</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28503</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.27715</span></span>
<span class="line"><span>Validation: Loss 0.26330 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.25558 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.25933</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24446</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23816</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22218</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21136</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.20513</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19449</span></span>
<span class="line"><span>Validation: Loss 0.18418 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.18021 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.18685</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17398</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16269</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15676</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15094</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14448</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13825</span></span>
<span class="line"><span>Validation: Loss 0.13205 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12968 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.13188</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12423</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11921</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11399</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11141</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10388</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10053</span></span>
<span class="line"><span>Validation: Loss 0.09668 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09441 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09645</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09341</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08706</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08427</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07853</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07731</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07539</span></span>
<span class="line"><span>Validation: Loss 0.07177 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06936 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.07208</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06811</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06402</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06315</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06022</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05685</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05581</span></span>
<span class="line"><span>Validation: Loss 0.05382 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05138 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05303</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04972</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04971</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04608</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04619</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04372</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04065</span></span>
<span class="line"><span>Validation: Loss 0.04064 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03831 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03994</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03905</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03626</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03557</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03459</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03245</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03142</span></span>
<span class="line"><span>Validation: Loss 0.03109 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02895 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02989</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02932</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02913</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02656</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02675</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02568</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02588</span></span>
<span class="line"><span>Validation: Loss 0.02445 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02256 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02320</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02348</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02334</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02089</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02141</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02055</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02094</span></span>
<span class="line"><span>Validation: Loss 0.01988 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01823 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01896</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01900</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01838</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01825</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01677</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01746</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01850</span></span>
<span class="line"><span>Validation: Loss 0.01667 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01524 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01651</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01649</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01468</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01574</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01445</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01488</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01347</span></span>
<span class="line"><span>Validation: Loss 0.01435 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01311 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01409</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01360</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01321</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01372</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01361</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01238</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01140</span></span>
<span class="line"><span>Validation: Loss 0.01264 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01153 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01231</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01141</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01113</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01168</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01257</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01181</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01183</span></span>
<span class="line"><span>Validation: Loss 0.01131 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01030 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01037</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01116</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01082</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01126</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00960</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01064</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01008</span></span>
<span class="line"><span>Validation: Loss 0.01022 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00930 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00928</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01021</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00961</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00986</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00917</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00976</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00918</span></span>
<span class="line"><span>Validation: Loss 0.00932 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00847 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00893</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00935</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00838</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00863</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00883</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00874</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00866</span></span>
<span class="line"><span>Validation: Loss 0.00856 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00777 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00809</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00839</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00795</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00804</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00781</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00823</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00850</span></span>
<span class="line"><span>Validation: Loss 0.00790 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00717 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00776</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00795</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00794</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00724</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00736</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00671</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00761</span></span>
<span class="line"><span>Validation: Loss 0.00733 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00664 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00753</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00709</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00672</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00707</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00679</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00667</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00652</span></span>
<span class="line"><span>Validation: Loss 0.00683 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00618 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00679</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00617</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00664</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00624</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00654</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00641</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00709</span></span>
<span class="line"><span>Validation: Loss 0.00638 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00578 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00632</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00612</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00605</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00613</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00601</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00591</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00568</span></span>
<span class="line"><span>Validation: Loss 0.00599 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00541 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00619</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00593</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00563</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00576</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00545</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00550</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00483</span></span>
<span class="line"><span>Validation: Loss 0.00563 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00509 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00586</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00526</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00533</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00538</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00505</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00555</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00455</span></span>
<span class="line"><span>Validation: Loss 0.00531 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00480 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.56054</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51134</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47898</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.46836</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.42067</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.40507</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.38537</span></span>
<span class="line"><span>Validation: Loss 0.37007 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36737 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37582</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.35528</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.34610</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.31575</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.30420</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28659</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.26095</span></span>
<span class="line"><span>Validation: Loss 0.26016 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.25870 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.26240</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24746</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23738</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22645</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21215</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.20102</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.18910</span></span>
<span class="line"><span>Validation: Loss 0.18259 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.18179 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.18371</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17617</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16711</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15867</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15149</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14368</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.12961</span></span>
<span class="line"><span>Validation: Loss 0.13100 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.13045 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.13234</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12591</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12052</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11420</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11125</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10359</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09782</span></span>
<span class="line"><span>Validation: Loss 0.09584 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09527 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09709</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09152</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08949</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08475</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07936</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07749</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07694</span></span>
<span class="line"><span>Validation: Loss 0.07106 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.07046 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.07171</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06842</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06513</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06145</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06147</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05832</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06033</span></span>
<span class="line"><span>Validation: Loss 0.05319 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05261 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05344</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05142</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04887</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04858</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04545</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04273</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04416</span></span>
<span class="line"><span>Validation: Loss 0.04003 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03951 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03993</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03871</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03547</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03641</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03497</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03479</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03054</span></span>
<span class="line"><span>Validation: Loss 0.03053 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03007 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03127</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02816</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02890</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02626</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02771</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02658</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02680</span></span>
<span class="line"><span>Validation: Loss 0.02396 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02354 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02562</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02447</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02148</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02063</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02054</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02222</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01905</span></span>
<span class="line"><span>Validation: Loss 0.01945 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01910 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01958</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01858</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01907</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01764</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01768</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01796</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01723</span></span>
<span class="line"><span>Validation: Loss 0.01632 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01602 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01695</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01718</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01679</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01493</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01408</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01369</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01566</span></span>
<span class="line"><span>Validation: Loss 0.01406 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01379 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01461</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01430</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01412</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01328</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01246</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01265</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01305</span></span>
<span class="line"><span>Validation: Loss 0.01237 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01213 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01212</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01294</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01207</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01248</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01162</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01105</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01066</span></span>
<span class="line"><span>Validation: Loss 0.01105 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01083 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01108</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01044</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01081</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01099</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01081</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01064</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00990</span></span>
<span class="line"><span>Validation: Loss 0.00999 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00980 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01048</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00956</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00936</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00907</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00994</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01042</span></span>
<span class="line"><span>Validation: Loss 0.00911 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00892 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00946</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00838</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00877</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00919</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00902</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00890</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00826</span></span>
<span class="line"><span>Validation: Loss 0.00836 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00819 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00897</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00896</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00809</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00771</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00806</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00790</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00665</span></span>
<span class="line"><span>Validation: Loss 0.00771 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00755 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00748</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00752</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00722</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00789</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00803</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00744</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00754</span></span>
<span class="line"><span>Validation: Loss 0.00716 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00702 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00699</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00705</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00713</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00668</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00788</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00681</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00637</span></span>
<span class="line"><span>Validation: Loss 0.00667 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00653 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00700</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00665</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00676</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00639</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00613</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00657</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00673</span></span>
<span class="line"><span>Validation: Loss 0.00624 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00611 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00638</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00621</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00628</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00587</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00651</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00602</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00516</span></span>
<span class="line"><span>Validation: Loss 0.00585 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00572 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00576</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00599</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00591</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00561</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00570</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00583</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00561</span></span>
<span class="line"><span>Validation: Loss 0.00550 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00539 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00551</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00524</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00578</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00572</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00547</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00511</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00507</span></span>
<span class="line"><span>Validation: Loss 0.00519 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00508 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {compress </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">} ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
<span class="line"><span> :ps_trained</span></span>
<span class="line"><span> :st_trained</span></span></code></pre></div><h2 id="Appendix" tabindex="-1">Appendix <a class="header-anchor" href="#Appendix" aria-label="Permalink to &quot;Appendix {#Appendix}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Julia Version 1.10.3</span></span>
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
<span class="line"><span>CUDA runtime 12.5, artifact installation</span></span>
<span class="line"><span>CUDA driver 12.5</span></span>
<span class="line"><span>NVIDIA driver 550.54.15, originally for CUDA 12.4</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA libraries: </span></span>
<span class="line"><span>- CUBLAS: 12.5.2</span></span>
<span class="line"><span>- CURAND: 10.3.6</span></span>
<span class="line"><span>- CUFFT: 11.2.3</span></span>
<span class="line"><span>- CUSOLVER: 11.6.2</span></span>
<span class="line"><span>- CUSPARSE: 12.4.1</span></span>
<span class="line"><span>- CUPTI: 23.0.0</span></span>
<span class="line"><span>- NVML: 12.0.0+550.54.15</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Julia packages: </span></span>
<span class="line"><span>- CUDA: 5.4.2</span></span>
<span class="line"><span>- CUDA_Driver_jll: 0.9.0+0</span></span>
<span class="line"><span>- CUDA_Runtime_jll: 0.14.0+1</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Toolchain:</span></span>
<span class="line"><span>- Julia: 1.10.3</span></span>
<span class="line"><span>- LLVM: 15.0.7</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>- JULIA_CUDA_HARD_MEMORY_LIMIT: 100%</span></span>
<span class="line"><span></span></span>
<span class="line"><span>1 device:</span></span>
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.303 GiB / 4.750 GiB available)</span></span>
<span class="line"><span>┌ Warning: LuxAMDGPU is loaded but the AMDGPU is not functional.</span></span>
<span class="line"><span>└ @ LuxAMDGPU ~/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/LuxAMDGPU/DaGeB/src/LuxAMDGPU.jl:19</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45),h=[l];function e(t,k,c,o,E,r){return n(),a("div",null,h)}const g=s(p,[["render",e]]);export{y as __pageData,g as default};
