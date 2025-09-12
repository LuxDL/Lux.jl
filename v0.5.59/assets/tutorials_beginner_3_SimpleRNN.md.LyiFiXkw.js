import{_ as s,c as a,o as n,a3 as i}from"./chunks/framework.Crd-7bEQ.js";const y=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),p={name:"tutorials/beginner/3_SimpleRNN.md"},l=i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, AMDGPU, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random,</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractExplicitContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/v0.5.59/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">:</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">       Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">AbstractExplicitContainerLayer{(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/v0.5.59/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/v0.5.59/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/v0.5.59/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.56246</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51233</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47545</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.45302</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.41811</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.40466</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.38951</span></span>
<span class="line"><span>Validation: Loss 0.36205 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36880 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36594</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.35766</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.33611</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.31753</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.29786</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.27794</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28364</span></span>
<span class="line"><span>Validation: Loss 0.25408 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.25780 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.25783</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24112</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23363</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22121</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21039</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19819</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.18482</span></span>
<span class="line"><span>Validation: Loss 0.17806 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.17968 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17974</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17015</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16239</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15575</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14621</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14030</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13813</span></span>
<span class="line"><span>Validation: Loss 0.12689 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12806 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12764</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12179</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11548</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11193</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10735</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10182</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09635</span></span>
<span class="line"><span>Validation: Loss 0.09193 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09311 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09400</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08737</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08497</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08091</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07849</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07475</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07376</span></span>
<span class="line"><span>Validation: Loss 0.06747 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06878 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06816</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06512</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06474</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06085</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05679</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05636</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05082</span></span>
<span class="line"><span>Validation: Loss 0.05010 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05141 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05132</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04868</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04750</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04480</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04467</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04066</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04268</span></span>
<span class="line"><span>Validation: Loss 0.03752 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03879 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03814</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03907</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03683</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03314</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03369</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.02999</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03054</span></span>
<span class="line"><span>Validation: Loss 0.02855 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02972 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02883</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02840</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02714</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02742</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02664</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02401</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02565</span></span>
<span class="line"><span>Validation: Loss 0.02239 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02344 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02364</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02332</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02128</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02082</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02133</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01996</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01794</span></span>
<span class="line"><span>Validation: Loss 0.01819 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01911 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01864</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01913</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01737</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01849</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01686</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01624</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01799</span></span>
<span class="line"><span>Validation: Loss 0.01529 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01609 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01520</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01672</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01503</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01443</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01492</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01516</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01253</span></span>
<span class="line"><span>Validation: Loss 0.01320 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01391 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01394</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01361</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01295</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01329</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01295</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01265</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01236</span></span>
<span class="line"><span>Validation: Loss 0.01164 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01228 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01245</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01139</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01197</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01191</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01177</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01074</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01198</span></span>
<span class="line"><span>Validation: Loss 0.01043 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01101 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01068</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01040</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01046</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01044</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01085</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01033</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01088</span></span>
<span class="line"><span>Validation: Loss 0.00944 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00997 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01014</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00963</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00987</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01013</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00898</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00878</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00937</span></span>
<span class="line"><span>Validation: Loss 0.00861 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00910 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00893</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00882</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00915</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00875</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00840</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00907</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00667</span></span>
<span class="line"><span>Validation: Loss 0.00791 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00836 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00845</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00817</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00819</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00819</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00767</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00781</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00801</span></span>
<span class="line"><span>Validation: Loss 0.00731 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00774 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00695</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00742</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00767</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00713</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00776</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00813</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00672</span></span>
<span class="line"><span>Validation: Loss 0.00679 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00719 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00727</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00684</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00737</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00696</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00668</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00670</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00689</span></span>
<span class="line"><span>Validation: Loss 0.00633 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00671 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00649</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00661</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00635</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00663</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00678</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00646</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00531</span></span>
<span class="line"><span>Validation: Loss 0.00592 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00628 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00653</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00671</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00599</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00618</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00562</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00584</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00497</span></span>
<span class="line"><span>Validation: Loss 0.00555 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00589 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00605</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00616</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00582</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00541</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00560</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00561</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00471</span></span>
<span class="line"><span>Validation: Loss 0.00523 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00555 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00536</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00485</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00562</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00575</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00516</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00555</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00584</span></span>
<span class="line"><span>Validation: Loss 0.00494 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00524 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.56154</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51100</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47399</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.44733</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.42533</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.40969</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.40569</span></span>
<span class="line"><span>Validation: Loss 0.37894 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.35906 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36518</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.35005</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.33300</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.31374</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.30731</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28726</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28238</span></span>
<span class="line"><span>Validation: Loss 0.26457 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.25325 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.26180</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24214</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23596</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22519</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.20937</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19658</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19031</span></span>
<span class="line"><span>Validation: Loss 0.18376 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.17897 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.18156</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17077</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16354</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15629</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14886</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14396</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13662</span></span>
<span class="line"><span>Validation: Loss 0.13157 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12830 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12905</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12488</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11871</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11352</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10684</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10288</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10001</span></span>
<span class="line"><span>Validation: Loss 0.09642 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09312 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09377</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09178</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08604</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08222</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07894</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07633</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07710</span></span>
<span class="line"><span>Validation: Loss 0.07186 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06827 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.07014</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06661</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06468</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06104</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06037</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05548</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05586</span></span>
<span class="line"><span>Validation: Loss 0.05410 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05058 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05203</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05026</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05090</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04580</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04301</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04238</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.03811</span></span>
<span class="line"><span>Validation: Loss 0.04105 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03771 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03866</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03766</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03653</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03558</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03427</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03145</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03190</span></span>
<span class="line"><span>Validation: Loss 0.03164 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02849 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03028</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02983</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02737</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02671</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02619</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02517</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02314</span></span>
<span class="line"><span>Validation: Loss 0.02506 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02220 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02272</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02146</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02348</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02174</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02210</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02002</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01927</span></span>
<span class="line"><span>Validation: Loss 0.02050 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01797 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01820</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01775</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01807</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01835</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01736</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01824</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01704</span></span>
<span class="line"><span>Validation: Loss 0.01726 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01506 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01633</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01642</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01504</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01476</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01507</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01403</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01510</span></span>
<span class="line"><span>Validation: Loss 0.01492 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01297 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01350</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01392</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01347</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01409</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01343</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01197</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01036</span></span>
<span class="line"><span>Validation: Loss 0.01313 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01141 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01233</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01133</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01201</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01106</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01175</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01178</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01280</span></span>
<span class="line"><span>Validation: Loss 0.01177 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01020 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01074</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01110</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01102</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01063</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00985</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01019</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01013</span></span>
<span class="line"><span>Validation: Loss 0.01065 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00921 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00965</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00933</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00971</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00979</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01019</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00903</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00892</span></span>
<span class="line"><span>Validation: Loss 0.00972 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00840 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00946</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00956</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00810</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00801</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00910</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00849</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00844</span></span>
<span class="line"><span>Validation: Loss 0.00895 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00771 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00804</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00811</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00810</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00769</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00816</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00797</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00949</span></span>
<span class="line"><span>Validation: Loss 0.00826 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00711 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00759</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00794</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00748</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00695</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00749</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00748</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00698</span></span>
<span class="line"><span>Validation: Loss 0.00766 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00659 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00727</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00725</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00679</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00692</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00686</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00665</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00650</span></span>
<span class="line"><span>Validation: Loss 0.00714 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00614 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00664</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00635</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00662</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00633</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00646</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00636</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00673</span></span>
<span class="line"><span>Validation: Loss 0.00668 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00574 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00644</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00593</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00621</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00592</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00572</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00631</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00535</span></span>
<span class="line"><span>Validation: Loss 0.00627 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00538 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00595</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00579</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00564</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00555</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00555</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00560</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00591</span></span>
<span class="line"><span>Validation: Loss 0.00590 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00506 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00529</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00538</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00558</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00545</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00528</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00509</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00573</span></span>
<span class="line"><span>Validation: Loss 0.00557 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00477 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {compress </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">} ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.484 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45),h=[l];function e(t,c,k,o,E,r){return n(),a("div",null,h)}const g=s(p,[["render",e]]);export{y as __pageData,g as default};
