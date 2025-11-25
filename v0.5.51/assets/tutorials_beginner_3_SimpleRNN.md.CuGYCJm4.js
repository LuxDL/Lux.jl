import{_ as s,c as a,o as n,a3 as i}from"./chunks/framework.DAcZ3hGF.js";const y=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),p={name:"tutorials/beginner/3_SimpleRNN.md"},l=i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxAMDGPU, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random,</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractExplicitContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/v0.5.51/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">:</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">       Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">AbstractExplicitContainerLayer{(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/v0.5.51/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/v0.5.51/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/v0.5.51/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>Epoch [  1]: Loss 0.56071</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50739</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47221</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.44208</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.43221</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.41238</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.38536</span></span>
<span class="line"><span>Validation: Loss 0.36530 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.35849 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37210</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.35331</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.33848</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.31696</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.29372</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28576</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.26538</span></span>
<span class="line"><span>Validation: Loss 0.25577 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.25209 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.25824</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24531</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23323</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21706</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.20896</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.20042</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.18869</span></span>
<span class="line"><span>Validation: Loss 0.17857 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.17678 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.18026</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16861</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16442</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15539</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14698</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14072</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13361</span></span>
<span class="line"><span>Validation: Loss 0.12728 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12619 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12802</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12074</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11655</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11198</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10751</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10121</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09850</span></span>
<span class="line"><span>Validation: Loss 0.09233 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09126 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09318</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08823</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08686</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07904</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07887</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07551</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07037</span></span>
<span class="line"><span>Validation: Loss 0.06793 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06679 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06786</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06591</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06432</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05947</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05808</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05664</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.04973</span></span>
<span class="line"><span>Validation: Loss 0.05054 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04937 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05158</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05092</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04809</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04333</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04037</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04321</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04192</span></span>
<span class="line"><span>Validation: Loss 0.03796 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03680 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03753</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03785</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03377</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03376</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03464</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03264</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03065</span></span>
<span class="line"><span>Validation: Loss 0.02898 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02789 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03032</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02796</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02720</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02631</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02606</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02513</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02148</span></span>
<span class="line"><span>Validation: Loss 0.02276 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02180 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02400</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02313</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02180</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02099</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02016</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01940</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01925</span></span>
<span class="line"><span>Validation: Loss 0.01852 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01769 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01894</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01855</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01841</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01806</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01656</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01591</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01705</span></span>
<span class="line"><span>Validation: Loss 0.01559 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01486 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01552</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01626</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01471</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01392</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01507</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01473</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01502</span></span>
<span class="line"><span>Validation: Loss 0.01343 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01280 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01389</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01290</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01294</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01249</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01354</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01303</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01186</span></span>
<span class="line"><span>Validation: Loss 0.01184 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01127 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01211</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01190</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01128</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01199</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01156</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01093</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01092</span></span>
<span class="line"><span>Validation: Loss 0.01059 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01008 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01105</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01020</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01001</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01065</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01079</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00957</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01153</span></span>
<span class="line"><span>Validation: Loss 0.00958 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00911 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00993</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00933</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00981</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00972</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00907</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00890</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00952</span></span>
<span class="line"><span>Validation: Loss 0.00873 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00831 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00848</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00922</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00890</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00899</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00819</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00836</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00776</span></span>
<span class="line"><span>Validation: Loss 0.00802 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00763 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00791</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00852</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00800</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00764</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00845</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00732</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00781</span></span>
<span class="line"><span>Validation: Loss 0.00741 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00704 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00801</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00725</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00735</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00738</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00731</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00707</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00685</span></span>
<span class="line"><span>Validation: Loss 0.00687 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00653 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00690</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00690</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00687</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00683</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00682</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00692</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00639</span></span>
<span class="line"><span>Validation: Loss 0.00640 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00608 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00656</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00666</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00638</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00651</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00627</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00594</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00671</span></span>
<span class="line"><span>Validation: Loss 0.00599 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00569 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00627</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00606</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00556</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00646</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00589</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00568</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00611</span></span>
<span class="line"><span>Validation: Loss 0.00562 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00533 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00582</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00559</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00577</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00537</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00568</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00575</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00476</span></span>
<span class="line"><span>Validation: Loss 0.00529 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00502 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00544</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00540</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00509</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00549</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00517</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00537</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00466</span></span>
<span class="line"><span>Validation: Loss 0.00499 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00473 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>Epoch [  1]: Loss 0.56142</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50780</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47336</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.44332</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.42759</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.40482</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.40112</span></span>
<span class="line"><span>Validation: Loss 0.37054 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36954 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36841</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.34780</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.33049</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.31733</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.30101</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28449</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.27376</span></span>
<span class="line"><span>Validation: Loss 0.25900 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.25840 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.25031</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24261</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23143</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21870</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21252</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.20023</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19531</span></span>
<span class="line"><span>Validation: Loss 0.18029 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.18002 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17984</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17066</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16170</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15387</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14566</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14196</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13645</span></span>
<span class="line"><span>Validation: Loss 0.12831 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12823 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12664</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12174</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11716</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11190</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10603</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10185</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09629</span></span>
<span class="line"><span>Validation: Loss 0.09335 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09329 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09308</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08867</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08611</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08032</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07834</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07421</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.06987</span></span>
<span class="line"><span>Validation: Loss 0.06899 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06890 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06878</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06557</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06168</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06049</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05805</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05524</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05506</span></span>
<span class="line"><span>Validation: Loss 0.05163 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05151 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05059</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04914</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04848</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04442</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04301</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04078</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04210</span></span>
<span class="line"><span>Validation: Loss 0.03897 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03885 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03749</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03795</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03642</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03265</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03136</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03239</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03452</span></span>
<span class="line"><span>Validation: Loss 0.02988 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02975 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02887</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02713</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02640</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02696</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02571</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02601</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02450</span></span>
<span class="line"><span>Validation: Loss 0.02356 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02343 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02331</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02246</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02130</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02098</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02040</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01970</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02038</span></span>
<span class="line"><span>Validation: Loss 0.01920 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01908 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01775</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01906</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01742</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01764</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01722</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01693</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01488</span></span>
<span class="line"><span>Validation: Loss 0.01615 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01605 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01554</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01529</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01443</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01587</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01421</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01479</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01248</span></span>
<span class="line"><span>Validation: Loss 0.01396 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01387 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01372</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01287</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01313</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01327</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01312</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01200</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01260</span></span>
<span class="line"><span>Validation: Loss 0.01232 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01224 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01185</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01146</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01186</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01144</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01108</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01156</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01135</span></span>
<span class="line"><span>Validation: Loss 0.01104 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01097 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01028</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01072</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01007</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01083</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01030</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00986</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01109</span></span>
<span class="line"><span>Validation: Loss 0.01000 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00992 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00936</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00986</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00938</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00922</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00917</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00968</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00883</span></span>
<span class="line"><span>Validation: Loss 0.00912 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00906 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00881</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00836</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00932</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00895</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00797</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00832</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00857</span></span>
<span class="line"><span>Validation: Loss 0.00839 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00832 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00841</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00914</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00755</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00816</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00765</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00706</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00679</span></span>
<span class="line"><span>Validation: Loss 0.00775 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00769 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00744</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00776</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00764</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00735</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00706</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00693</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00709</span></span>
<span class="line"><span>Validation: Loss 0.00720 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00714 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00704</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00742</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00667</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00679</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00672</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00645</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00662</span></span>
<span class="line"><span>Validation: Loss 0.00672 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00667 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00616</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00665</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00652</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00630</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00603</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00656</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00665</span></span>
<span class="line"><span>Validation: Loss 0.00629 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00624 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00642</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00595</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00580</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00630</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00576</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00575</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00564</span></span>
<span class="line"><span>Validation: Loss 0.00590 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00585 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00577</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00554</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00573</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00569</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00551</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00547</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00567</span></span>
<span class="line"><span>Validation: Loss 0.00555 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00551 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00527</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00528</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00510</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00548</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00515</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00544</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00551</span></span>
<span class="line"><span>Validation: Loss 0.00524 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00520 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {compress </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">} ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.453 GiB / 4.750 GiB available)</span></span>
<span class="line"><span>┌ Warning: LuxAMDGPU is loaded but the AMDGPU is not functional.</span></span>
<span class="line"><span>└ @ LuxAMDGPU ~/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6/packages/LuxAMDGPU/sGa0S/src/LuxAMDGPU.jl:19</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45),h=[l];function e(t,k,c,o,E,r){return n(),a("div",null,h)}const g=s(p,[["render",e]]);export{y as __pageData,g as default};
