import{_ as s,c as a,o as n,a3 as i}from"./chunks/framework.BH0Emp04.js";const y=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),p={name:"tutorials/beginner/3_SimpleRNN.md"},l=i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxAMDGPU, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random,</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractExplicitContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/v0.5.46/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">:</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">       Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">AbstractExplicitContainerLayer{(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/v0.5.46/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/v0.5.46/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/v0.5.46/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; lstm_cell, classifier) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray{T, 3}</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {T}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        x_init, x_rest </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Iterators</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">peel</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">_eachslice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        y, carry </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_init)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x_rest</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            y, carry </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x, carry))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vec</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y))</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Experimental</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">apply_gradients</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state, gs)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>Epoch [  1]: Loss 0.56322</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50963</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47135</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.44875</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.42638</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.41100</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.37006</span></span>
<span class="line"><span>Validation: Loss 0.36385 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.35502 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36675</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.35791</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.33520</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.31488</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.29437</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28376</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28472</span></span>
<span class="line"><span>Validation: Loss 0.25484 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.24970 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.25478</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24491</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22729</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22247</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21182</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19893</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19211</span></span>
<span class="line"><span>Validation: Loss 0.17831 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.17600 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17999</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16973</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16447</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15416</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14642</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14135</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13619</span></span>
<span class="line"><span>Validation: Loss 0.12741 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12612 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12766</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12257</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11715</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11127</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10637</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10177</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09959</span></span>
<span class="line"><span>Validation: Loss 0.09253 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09124 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09350</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08944</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08456</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08098</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07906</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07494</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07224</span></span>
<span class="line"><span>Validation: Loss 0.06810 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06664 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06907</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06665</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06064</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06045</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05817</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05731</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05501</span></span>
<span class="line"><span>Validation: Loss 0.05067 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04912 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05212</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05049</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04737</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04481</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04367</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04099</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.03955</span></span>
<span class="line"><span>Validation: Loss 0.03806 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03656 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03961</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03755</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03658</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03308</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03295</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03194</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03005</span></span>
<span class="line"><span>Validation: Loss 0.02901 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02758 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02917</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02861</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02798</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02569</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02654</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02488</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02617</span></span>
<span class="line"><span>Validation: Loss 0.02277 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02144 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02426</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02285</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02019</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02317</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02038</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01932</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01984</span></span>
<span class="line"><span>Validation: Loss 0.01850 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01734 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01786</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01809</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01797</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01789</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01718</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01792</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01710</span></span>
<span class="line"><span>Validation: Loss 0.01554 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01451 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01648</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01527</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01599</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01494</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01452</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01355</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01545</span></span>
<span class="line"><span>Validation: Loss 0.01341 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01251 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01369</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01317</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01336</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01365</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01255</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01279</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01253</span></span>
<span class="line"><span>Validation: Loss 0.01182 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01102 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01234</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01275</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01153</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01122</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01123</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01137</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01066</span></span>
<span class="line"><span>Validation: Loss 0.01058 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00985 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01104</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01080</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01034</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01026</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01038</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01042</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00994</span></span>
<span class="line"><span>Validation: Loss 0.00958 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00891 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01018</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00962</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00926</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00956</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00983</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00885</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00979</span></span>
<span class="line"><span>Validation: Loss 0.00874 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00813 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00855</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00890</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00923</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00853</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00883</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00847</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00878</span></span>
<span class="line"><span>Validation: Loss 0.00804 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00746 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00822</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00821</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00793</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00805</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00779</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00815</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00816</span></span>
<span class="line"><span>Validation: Loss 0.00743 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00689 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00808</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00733</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00756</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00708</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00727</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00752</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00735</span></span>
<span class="line"><span>Validation: Loss 0.00689 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00639 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00694</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00732</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00701</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00689</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00698</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00652</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00699</span></span>
<span class="line"><span>Validation: Loss 0.00643 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00596 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00644</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00712</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00592</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00662</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00662</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00631</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00604</span></span>
<span class="line"><span>Validation: Loss 0.00601 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00557 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00658</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00598</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00663</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00611</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00534</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00588</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00584</span></span>
<span class="line"><span>Validation: Loss 0.00564 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00522 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00555</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00568</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00581</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00575</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00571</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00565</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00614</span></span>
<span class="line"><span>Validation: Loss 0.00531 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00491 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00563</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00530</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00485</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00575</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00526</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00547</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00546</span></span>
<span class="line"><span>Validation: Loss 0.00501 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00463 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>Epoch [  1]: Loss 0.56062</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51265</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48013</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.44727</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.43144</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.40892</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.37540</span></span>
<span class="line"><span>Validation: Loss 0.36521 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36721 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36345</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.35259</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.33181</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.31620</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.30826</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28890</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28251</span></span>
<span class="line"><span>Validation: Loss 0.25608 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.25716 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.26370</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24031</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23538</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22149</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21139</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19993</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.18421</span></span>
<span class="line"><span>Validation: Loss 0.17932 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.17965 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.18115</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17157</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16311</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15638</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14717</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14250</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13539</span></span>
<span class="line"><span>Validation: Loss 0.12805 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12820 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12812</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12195</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11693</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11307</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10820</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10189</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10171</span></span>
<span class="line"><span>Validation: Loss 0.09291 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09307 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09364</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08891</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08587</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08214</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07834</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07527</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07663</span></span>
<span class="line"><span>Validation: Loss 0.06839 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06858 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06830</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06704</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06281</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06199</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05931</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05504</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05436</span></span>
<span class="line"><span>Validation: Loss 0.05091 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05111 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05308</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04989</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04783</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04513</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04404</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04087</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.03954</span></span>
<span class="line"><span>Validation: Loss 0.03819 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03839 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03857</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03777</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03578</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03460</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03375</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03203</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.02893</span></span>
<span class="line"><span>Validation: Loss 0.02905 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02926 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02912</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02720</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02920</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02653</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02543</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02584</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02450</span></span>
<span class="line"><span>Validation: Loss 0.02274 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02296 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02274</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02229</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02402</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02027</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02021</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02009</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02132</span></span>
<span class="line"><span>Validation: Loss 0.01845 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01865 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01882</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01879</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01842</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01783</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01725</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01652</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01405</span></span>
<span class="line"><span>Validation: Loss 0.01548 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01565 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01488</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01533</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01604</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01468</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01482</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01476</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01515</span></span>
<span class="line"><span>Validation: Loss 0.01336 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01352 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01325</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01422</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01288</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01338</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01302</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01261</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01127</span></span>
<span class="line"><span>Validation: Loss 0.01176 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01191 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01202</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01144</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01188</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01178</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01224</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01130</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.00885</span></span>
<span class="line"><span>Validation: Loss 0.01053 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01066 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01076</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01130</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01103</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01024</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00976</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01014</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00940</span></span>
<span class="line"><span>Validation: Loss 0.00953 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00965 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00968</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01030</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00961</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00903</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00902</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00972</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00885</span></span>
<span class="line"><span>Validation: Loss 0.00870 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00882 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00897</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00910</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00862</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00850</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00815</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00888</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00933</span></span>
<span class="line"><span>Validation: Loss 0.00799 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00810 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00807</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00748</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00810</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00827</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00803</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00834</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00772</span></span>
<span class="line"><span>Validation: Loss 0.00739 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00749 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00767</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00777</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00740</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00769</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00720</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00721</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00630</span></span>
<span class="line"><span>Validation: Loss 0.00685 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00695 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00682</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00725</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00669</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00715</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00660</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00708</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00664</span></span>
<span class="line"><span>Validation: Loss 0.00639 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00648 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00662</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00702</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00671</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00629</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00615</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00595</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00661</span></span>
<span class="line"><span>Validation: Loss 0.00597 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00606 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00643</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00588</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00621</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00606</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00578</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00596</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00599</span></span>
<span class="line"><span>Validation: Loss 0.00560 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00568 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00561</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00569</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00595</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00580</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00533</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00561</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00611</span></span>
<span class="line"><span>Validation: Loss 0.00527 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00535 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00539</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00576</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00555</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00544</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00469</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00510</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00611</span></span>
<span class="line"><span>Validation: Loss 0.00497 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00504 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {compress </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">} ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span>- CUDA: 5.3.3</span></span>
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
