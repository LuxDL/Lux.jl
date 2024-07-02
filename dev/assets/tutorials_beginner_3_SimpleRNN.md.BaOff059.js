import{_ as s,c as a,o as n,a3 as i}from"./chunks/framework.Br70pw-E.js";const y=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),p={name:"tutorials/beginner/3_SimpleRNN.md"},l=i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, AMDGPU, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random,</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractExplicitContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/dev/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">:</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">       Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">AbstractExplicitContainerLayer{(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/dev/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/dev/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/dev/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.56133</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51074</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47674</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.43951</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.43592</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.41654</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.38822</span></span>
<span class="line"><span>Validation: Loss 0.36264 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37747 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37422</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.34804</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.33213</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.32787</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.29510</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.29381</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.26546</span></span>
<span class="line"><span>Validation: Loss 0.25533 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.26378 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.25961</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24999</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23173</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22505</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21047</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.20060</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.18674</span></span>
<span class="line"><span>Validation: Loss 0.17951 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.18377 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.18192</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17093</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16555</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15724</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15097</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14198</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13641</span></span>
<span class="line"><span>Validation: Loss 0.12856 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.13146 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.13069</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12177</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11859</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11414</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10976</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10305</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10035</span></span>
<span class="line"><span>Validation: Loss 0.09347 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09620 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09425</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09112</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08685</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08523</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08009</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07470</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07215</span></span>
<span class="line"><span>Validation: Loss 0.06877 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.07150 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06915</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06784</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06538</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06248</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05884</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05640</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05469</span></span>
<span class="line"><span>Validation: Loss 0.05102 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05373 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05160</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05010</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04767</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04702</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04441</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04378</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04098</span></span>
<span class="line"><span>Validation: Loss 0.03810 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04068 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03850</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03799</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03594</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03641</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03337</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03147</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03607</span></span>
<span class="line"><span>Validation: Loss 0.02887 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03124 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02923</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02896</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02794</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02661</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02799</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02439</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02555</span></span>
<span class="line"><span>Validation: Loss 0.02255 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02463 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02354</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02229</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02157</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02273</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02161</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01996</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01864</span></span>
<span class="line"><span>Validation: Loss 0.01827 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02009 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01757</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01925</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01818</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01812</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01781</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01663</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01833</span></span>
<span class="line"><span>Validation: Loss 0.01531 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01689 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01614</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01655</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01526</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01598</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01333</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01459</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01373</span></span>
<span class="line"><span>Validation: Loss 0.01319 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01457 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01332</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01429</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01350</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01313</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01438</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01110</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01244</span></span>
<span class="line"><span>Validation: Loss 0.01161 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01284 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01211</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01252</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01122</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01129</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01160</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01142</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01276</span></span>
<span class="line"><span>Validation: Loss 0.01037 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01150 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01061</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01092</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01104</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01053</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01109</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00915</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01028</span></span>
<span class="line"><span>Validation: Loss 0.00937 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01039 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00965</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00983</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00980</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00886</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00983</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00944</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00939</span></span>
<span class="line"><span>Validation: Loss 0.00854 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00949 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00940</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00913</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00906</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00849</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00825</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00853</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00732</span></span>
<span class="line"><span>Validation: Loss 0.00783 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00872 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00831</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00872</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00800</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00787</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00729</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00805</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00822</span></span>
<span class="line"><span>Validation: Loss 0.00723 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00805 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00713</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00766</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00757</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00752</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00701</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00767</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00783</span></span>
<span class="line"><span>Validation: Loss 0.00671 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00748 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00714</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00696</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00730</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00713</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00650</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00678</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00588</span></span>
<span class="line"><span>Validation: Loss 0.00624 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00697 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00682</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00658</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00602</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00665</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00626</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00633</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00681</span></span>
<span class="line"><span>Validation: Loss 0.00584 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00652 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00622</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00621</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00610</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00627</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00600</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00561</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00550</span></span>
<span class="line"><span>Validation: Loss 0.00547 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00612 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00609</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00560</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00592</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00548</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00537</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00553</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00602</span></span>
<span class="line"><span>Validation: Loss 0.00515 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00576 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00545</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00537</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00535</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00564</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00563</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00469</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00516</span></span>
<span class="line"><span>Validation: Loss 0.00485 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00543 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.56172</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51444</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47867</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.45201</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.42375</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.40841</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.39319</span></span>
<span class="line"><span>Validation: Loss 0.36590 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37529 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36842</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.35469</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.33246</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.32306</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.30249</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28609</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.27319</span></span>
<span class="line"><span>Validation: Loss 0.25688 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.26187 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.25964</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24451</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23382</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21936</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21424</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19775</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19601</span></span>
<span class="line"><span>Validation: Loss 0.17974 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.18195 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.18052</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17182</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16467</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15725</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14727</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14142</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13653</span></span>
<span class="line"><span>Validation: Loss 0.12847 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12964 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12727</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12370</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11820</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11273</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10708</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10366</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09719</span></span>
<span class="line"><span>Validation: Loss 0.09321 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09450 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09485</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09031</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08632</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08142</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07796</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07503</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07203</span></span>
<span class="line"><span>Validation: Loss 0.06848 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.07000 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.07006</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06602</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06510</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05915</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05903</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05517</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05445</span></span>
<span class="line"><span>Validation: Loss 0.05087 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05249 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05188</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05092</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04952</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04445</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04358</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04038</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.03908</span></span>
<span class="line"><span>Validation: Loss 0.03812 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03969 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04021</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03630</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03637</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03584</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03196</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03159</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.02895</span></span>
<span class="line"><span>Validation: Loss 0.02900 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03048 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03014</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02827</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02736</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02844</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02540</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02415</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02217</span></span>
<span class="line"><span>Validation: Loss 0.02272 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02407 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02448</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02104</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02223</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02149</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02012</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02055</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01955</span></span>
<span class="line"><span>Validation: Loss 0.01846 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01964 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01868</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01835</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01788</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01852</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01750</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01591</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01674</span></span>
<span class="line"><span>Validation: Loss 0.01549 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01651 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01537</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01481</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01457</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01486</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01584</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01503</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01443</span></span>
<span class="line"><span>Validation: Loss 0.01336 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01426 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01366</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01341</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01349</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01176</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01324</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01308</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01299</span></span>
<span class="line"><span>Validation: Loss 0.01176 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01255 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01278</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01200</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01166</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01109</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01140</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01093</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01098</span></span>
<span class="line"><span>Validation: Loss 0.01051 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01122 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01068</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01036</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01026</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01079</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01020</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01045</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00961</span></span>
<span class="line"><span>Validation: Loss 0.00950 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01015 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00987</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01001</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00892</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01012</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00913</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00918</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00776</span></span>
<span class="line"><span>Validation: Loss 0.00867 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00926 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00859</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00879</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00922</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00886</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00876</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00797</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00780</span></span>
<span class="line"><span>Validation: Loss 0.00797 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00853 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00834</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00863</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00758</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00755</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00779</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00794</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00818</span></span>
<span class="line"><span>Validation: Loss 0.00736 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00787 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00782</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00760</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00698</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00696</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00754</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00730</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00777</span></span>
<span class="line"><span>Validation: Loss 0.00682 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00731 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00720</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00692</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00705</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00658</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00690</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00648</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00702</span></span>
<span class="line"><span>Validation: Loss 0.00636 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00681 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00686</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00647</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00703</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00616</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00608</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00596</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00582</span></span>
<span class="line"><span>Validation: Loss 0.00594 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00637 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00627</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00621</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00624</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00577</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00570</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00582</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00578</span></span>
<span class="line"><span>Validation: Loss 0.00557 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00597 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00597</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00584</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00599</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00530</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00554</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00534</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00476</span></span>
<span class="line"><span>Validation: Loss 0.00524 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00562 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00549</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00561</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00516</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00556</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00498</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00484</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00593</span></span>
<span class="line"><span>Validation: Loss 0.00494 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00530 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {compress </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">} ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
